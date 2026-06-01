from __future__ import annotations

import os
import time

import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Slonik-7B", page_icon="🐘", layout="wide")
st.title("🐘 Slonik-7B — PostgreSQL Text-to-SQL")

API = os.getenv("SLONIK_API", "http://localhost:8001")


with st.sidebar:
    st.header("Schema")
    schema_input = st.text_area(
        "DDL",
        height=320,
        value="""CREATE TABLE customers (
  id INT PRIMARY KEY,
  name TEXT,
  country TEXT,
  signed_up TIMESTAMPTZ,
  metadata JSONB
);

CREATE TABLE orders (
  id INT PRIMARY KEY,
  customer_id INT REFERENCES customers(id),
  total NUMERIC(10,2),
  created_at TIMESTAMPTZ,
  items JSONB
);""",
    )
    st.markdown("---")
    st.header("Connection (optional)")
    pg_host = st.text_input("Host", value=os.getenv("PG_HOST", "localhost"))
    pg_port = st.number_input("Port", value=int(os.getenv("PG_PORT", "5432")))
    pg_user = st.text_input("User", value=os.getenv("PG_USER", "slonik"))
    pg_pw = st.text_input("Password", type="password", value=os.getenv("PG_PASSWORD", ""))
    pg_db = st.text_input("Database", value=os.getenv("PG_DB", "postgres"))
    run_live = st.checkbox("Execute against this database", value=False)


col_q, col_out = st.columns([1, 1])

with col_q:
    question = st.text_area(
        "Question",
        value="Top 5 customers by total spend in Germany last quarter",
        height=120,
    )
    evidence = st.text_input("Hints (optional)")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    go = st.button("Generate SQL", type="primary", use_container_width=True)

with col_out:
    st.markdown("### Generated SQL")
    sql_box = st.empty()
    meta_box = st.empty()
    result_box = st.empty()


def _generate(schema: str, q: str, ev: str, temp: float) -> dict:
    r = httpx.post(
        f"{API}/generate",
        json={"schema": schema, "question": q, "evidence": ev, "temperature": temp},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def _run_pg(sql: str) -> pd.DataFrame:
    import psycopg
    conn_str = f"postgresql://{pg_user}:{pg_pw}@{pg_host}:{pg_port}/{pg_db}"
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d.name for d in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=cols)


if go:
    try:
        t0 = time.perf_counter()
        result = _generate(schema_input, question, evidence, temperature)
        sql_box.code(result["sql"], language="sql")
        meta_box.caption(
            f"latency: {result['latency_ms']} ms  |  "
            f"valid: {result.get('valid')}  |  "
            f"client total: {int((time.perf_counter() - t0) * 1000)} ms"
        )
        if result.get("parse_error"):
            st.warning(f"Parse error: {result['parse_error']}")

        if run_live and result.get("valid"):
            try:
                df = _run_pg(result["sql"])
                result_box.dataframe(df, use_container_width=True)
            except Exception as e:
                result_box.error(f"Execution failed: {e}")
    except httpx.HTTPError as e:
        st.error(f"API error: {e}")
