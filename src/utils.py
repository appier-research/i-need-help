import re

def parse_sql(plain_result: str):
    # # TODO: move this somewhere else
    pattern = r"```sql\n([\s\S]*?)\n```"
    if type(plain_result) == str:
        sql_block = plain_result
    else:
        sql_block = 'SELECT' + plain_result['choices'][0]['text']
    
    SQL_PREFIX = "```sql\n"
    SQL_SUFFIX = "```"
    if (sql_block[:len(SQL_PREFIX)] == SQL_PREFIX) and (sql_block[-len(SQL_SUFFIX):] == SQL_SUFFIX):
        sql_block = sql_block[len(SQL_PREFIX):-len(SQL_SUFFIX)]
    elif '```' in sql_block:
        match = re.search(pattern, plain_result)
        if match:
            sql_block = match.group(1)
        else:
            sql_block = sql_block.replace('```sql\n', '```').replace('```SQL\n', '```')
            sql_block = sql_block.split('```', maxsplit=1)[-1]
            sql_block = sql_block.replace('```', '').strip()
    sql_block = sql_block.strip()
    if 'SELECT' != sql_block[:len('SELECT')]:
        sql_block = 'SELECT ' + sql_block
    return sql_block
