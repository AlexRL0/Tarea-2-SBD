import hashlib
import io
import os
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

S3_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("MINIO_ACCESS_KEY", "admin")
AWS_SECRET_ACCESS_KEY = os.getenv("MINIO_SECRET_KEY", "13082004A")
SALT = os.getenv("RGPD_SALT", "ELECTRONOVA_RGPD_SALT_2026_")

BUCKET_BRONZE = "bronze-raw"
KEY_BRONZE = "ventas/ventas_raw.csv"
BUCKET_SILVER = "silver-curated"
PREFIX_SILVER = "ventas"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    dt_iso = pd.to_datetime(cleaned, format="%Y-%m-%d", errors="coerce")
    dt_eu = pd.to_datetime(cleaned, format="%d/%m/%Y", errors="coerce")
    result = dt_iso.fillna(dt_eu)
    missing = result.isna()
    if missing.any():
        result.loc[missing] = pd.to_datetime(cleaned.loc[missing], errors="coerce")
    return result.dt.date


def hash_email(val) -> str | None:
    if pd.isna(val) or str(val).strip() == "":
        return None
    normalized_email = str(val).strip().lower()
    return hashlib.sha256(f"{SALT}{normalized_email}".encode("utf-8")).hexdigest()


def main():
    s3 = get_s3_client()

    obj = s3.get_object(Bucket=BUCKET_BRONZE, Key=KEY_BRONZE)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str)
    filas_leidas = len(df)

    df.columns = [c.strip() for c in df.columns]

    for col in [
        "id_pedido",
        "sku",
        "producto",
        "categoria",
        "ciudad",
        "canal_venta",
        "metodo_pago",
        "estado",
        "email_cliente",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    if "id_pedido" in df.columns:
        df = df[df["id_pedido"].notna() & (df["id_pedido"] != "")]
        df = df.drop_duplicates(subset=["id_pedido"], keep="first")

    df["fecha_pedido"] = parse_mixed_dates(df["fecha_pedido"])
    df = df[df["fecha_pedido"].notna()]
    df["anio"] = pd.to_datetime(df["fecha_pedido"]).dt.year.astype(int)

    df["ciudad"] = (
        df["ciudad"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
        .str.title()
    )
    df.loc[df["ciudad"].isin(["", "Nan", "None"]), "ciudad"] = pd.NA
    df = df[df["ciudad"].notna()]

    df["categoria"] = df["categoria"].astype(str).str.strip()
    df = df[
        df["categoria"].notna()
        & (df["categoria"] != "")
        & (df["categoria"].str.lower() != "nan")
    ]

    df["unidades"] = pd.to_numeric(df["unidades"], errors="coerce")
    df["precio_unitario"] = pd.to_numeric(df["precio_unitario"], errors="coerce")

    df = df[
        df["unidades"].notna()
        & (df["unidades"] > 0)
        & df["precio_unitario"].notna()
        & (df["precio_unitario"] > 0)
    ].copy()

    df["unidades"] = df["unidades"].astype(int)
    df["precio_unitario"] = df["precio_unitario"].astype(float)
    df["importe_total"] = (df["unidades"] * df["precio_unitario"]).round(2)

    df["cliente_hash"] = df["email_cliente"].apply(hash_email)
    df = df.drop(columns=["email_cliente"])

    filas_cargadas = len(df)
    filas_descartadas = filas_leidas - filas_cargadas

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_SILVER, Prefix=f"{PREFIX_SILVER}/"):
        contents = page.get("Contents", [])
        if contents:
            s3.delete_objects(
                Bucket=BUCKET_SILVER,
                Delete={"Objects": [{"Key": o["Key"]} for o in contents]},
            )

    for (anio_val, cat_val), group_df in df.groupby(["anio", "categoria"], dropna=False):
        safe_cat = str(cat_val).strip().replace("/", "-")
        part_table = pa.Table.from_pandas(group_df, preserve_index=False)
        buf = io.BytesIO()
        pq.write_table(part_table, buf, compression="snappy")
        buf.seek(0)
        s3_key = f"{PREFIX_SILVER}/anio={anio_val}/categoria={safe_cat}/data.parquet"
        s3.put_object(Bucket=BUCKET_SILVER, Key=s3_key, Body=buf.getvalue())

    print("=== RESUMEN ETL BRONZE -> SILVER ===")
    print(f"Filas leídas:      {filas_leidas}")
    print(f"Filas descartadas: {filas_descartadas}")
    print(f"Filas cargadas:     {filas_cargadas}")
    print(f"Destino S3:         s3://{BUCKET_SILVER}/{PREFIX_SILVER}/")


if __name__ == "__main__":
    main()