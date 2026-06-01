from __future__ import annotations


def validate_postgres(sql: str) -> tuple[bool, str | None]:
    try:
        import pglast
    except ImportError:
        try:
            import sqlglot
            sqlglot.parse_one(sql, read="postgres")
            return True, None
        except Exception as e:
            return False, str(e)

    try:
        pglast.parse_sql(sql)
        return True, None
    except pglast.parser.ParseError as e:
        return False, str(e)


def fingerprint(sql: str) -> str | None:
    try:
        import pglast
        return pglast.parser.fingerprint(sql)
    except Exception:
        return None
