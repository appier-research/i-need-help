import os
import re
import random
import sqlite3
import textwrap
from llms import LLM
from sql_execution import execute_model

class Ask():
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path

    def __call__(
        self, 
        db_schema: str,
        question: str,
        gen_sql: str,
        db_id: str = None
    ) -> str:
        raise NotImplementedError

class DirectAsk(Ask):
    name = "DA"
    PROMPT_TEMPLATE = textwrap.dedent(f"""\
    You are currently doing the text-to-SQL task. Based on the information provided (Database schema, User's question), you have to determine whether additional hints are required for you to generate the SQL correctly to answer the user's question. You should only ask for additional hints when you actually need them, since you will also be evaluated based on the number of times you ask for hints, which would be provided by the user.

    information provided (enclosed by triple backticks):
    ```
    Database schema: {{db_schema}}
    User's question: {{question}}
    ```

    Answer a single word Yes if you need hints (since the information provided is not enough to generate SQL correctly). Answer a single word No if hints are not required (since you are already confident to generate SQL).
    Do you need additional hints? Answer (Yes / No):""")

    def __init__(self, agent: LLM, database_path) -> None:
        super().__init__(database_path)
        self.agent = agent

    def __call__(
        self,
        db_schema: str,
        question: str,
        gen_sql: str = None,
        db_id: str = None
    ) -> dict:
        prompt = self.PROMPT_TEMPLATE.format(db_schema=db_schema, question=question)
        res_text, logprobs = self.agent(prompt, max_tokens=1, top_logprobs=20)
        need_hint = -1
        if "No" in res_text:
            need_hint = 0
        elif "Yes" in res_text:
            need_hint = 1
        return {
            "need_hint": need_hint,
            "logprobs": logprobs,
            "hint_prompt": prompt,
            "hint_response": res_text
        }

class WriteThenAsk(Ask):
    name = "WA"
    PROMPT_TEMPLATE = textwrap.dedent(f"""\
    You are currently doing the text-to-SQL task. Based on the information provided (Database schema, User's question, Originally generated SQL), you have to determine whether additional hints are required for you to re-generate the SQL again to correctly answer the user's question. You should only ask for additional hints when you actually need them, since you will also be evaluated based on the number of times you ask for hints, which would be provided by the user.

    information provided (enclosed by triple backticks):
    ```
    Database schema: {{db_schema}}
    User's question: {{question}}
    Originally generated SQL: {{gen_sql}}
    ```

    Answer a single word Yes if you need hints (since the originally generated SQL cannot answer the user's question correctly). Answer a single word No if hints are not required (since you are already confident about the originally generated SQL).
    Do you need additional hints? Answer (Yes / No):""")

    def __init__(self, agent: LLM, database_path) -> None:
        super().__init__(database_path)
        self.agent = agent

    def __call__(
        self,
        db_schema: str,
        question: str,
        gen_sql: str,
        db_id: str = None
    ) -> dict:
        prompt = self.PROMPT_TEMPLATE.format(db_schema=db_schema, question=question, gen_sql=gen_sql)
        res_text, logprobs = self.agent(prompt, max_tokens=1, top_logprobs=20)
        need_hint = -1
        if "No" in res_text:
            need_hint = 0
        elif "Yes" in res_text:
            need_hint = 1
        return {
            "need_hint": need_hint,
            "logprobs": logprobs,
            "hint_prompt": prompt,
            "hint_response": res_text
        }

class ExecuteThenAsk(Ask):
    name = "EA"
    PROMPT_TEMPLATE = textwrap.dedent(f"""\
    You are currently doing the text-to-SQL task. Based on the information provided (Database schema, User's question, Originally generated SQL, SQL execution results), you have to determine whether additional hints are required for you to re-generate the SQL again to correctly answer the user's question. You should only ask for additional hints when you actually need them, since you will also be evaluated based on the number of times you ask for hints, which would be provided by the user.

    information provided (enclosed by triple backticks):
    ```
    Database schema: {{db_schema}}
    User's question: {{question}}
    Originally generated SQL: {{gen_sql}}
    SQL execution results: {{exe_results}}
    ```

    Answer a single word Yes if you need hints (since the original SQL execution results cannot answer the user's question correctly). Answer a single word No if hints are not required (since you are already confident about the original SQL execution results).
    Do you need additional hints? Answer (Yes / No):""")

    def __init__(self, agent: LLM, database_path) -> None:
        super().__init__(database_path)
        self.agent = agent

    def __call__(
        self,
        db_schema: str,
        question: str,
        gen_sql: str,
        db_id: str
    ) -> dict:
        sqlite_file = os.path.join(self.database_path, db_id, db_id + ".sqlite")
        _, (exe_results, _) = execute_model(gen_sql, gen_sql, sqlite_file, meta_time_out=30)
        prompt = self.PROMPT_TEMPLATE.format(db_schema=db_schema, question=question, gen_sql=gen_sql, exe_results=exe_results)
        res_text, logprobs = self.agent(prompt, max_tokens=1, top_logprobs=20)
        need_hint = -1
        if "No" in res_text:
            need_hint = 0
        elif "Yes" in res_text:
            need_hint = 1
        return {
            "need_hint": need_hint,
            "logprobs": logprobs,
            "hint_prompt": prompt,
            "hint_response": res_text
        }

class ExecuteThenAskVerbalized(Ask):
    name = "EAV"
    PROMPT_TEMPLATE = textwrap.dedent(f"""\
    You are currently doing the text-to-SQL task. Based on the information provided (Database schema, User's question, Originally generated SQL, SQL execution results), you have to determine whether additional hints are required for you to re-generate the SQL again to correctly answer the user's question. You should only ask for additional hints when you actually need them, since you will also be evaluated based on the number of times you ask for hints, which would be provided by the user.

    information provided (enclosed by triple backticks):
    ```
    Database schema: {{db_schema}}
    User's question: {{question}}
    Originally generated SQL: {{gen_sql}}
    SQL execution results: {{exe_results}}
    ```

    Do you need additional hints? Provide the precise probability that you need hints (closer to 0 means you don't need hints, closer to 1 means you need hints).
    Give ONLY the precise probability to five decimal places (format: 0.abcde, where abcde can be different digits), no other words or explanations are needed.""")

    def __init__(self, agent: LLM, database_path) -> None:
        super().__init__(database_path)
        self.agent = agent

    def __call__(
        self,
        db_schema: str,
        question: str,
        gen_sql: str,
        db_id: str
    ) -> dict:
        sqlite_file = os.path.join(self.database_path, db_id, db_id + ".sqlite")
        _, (exe_results, _) = execute_model(gen_sql, gen_sql, sqlite_file, meta_time_out=30)
        prompt = self.PROMPT_TEMPLATE.format(db_schema=db_schema, question=question, gen_sql=gen_sql, exe_results=exe_results)
        res_text, logprobs = self.agent(prompt, max_tokens=1, top_logprobs=20)
        res_text = res_text.strip()
        # print(res_text)
        try:
            prob = float(res_text)
        except ValueError:  # parse the float value out of res_text
            float_text = re.search(r"\d+\.\d+", res_text)
            if float_text:
                print(f"float_text searched: {float_text.group()}")
                prob = float(float_text.group())
            else:
                print(f"float_text not found: {res_text}, using random value")
                prob = random.random()
        return {
            "need_hint": prob,
            "logprobs": logprobs,
            "hint_prompt": prompt,
            "hint_response": res_text
        }

def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]

    # Print the column names
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    # print(header)
    # Print the values
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output


def generate_schema_prompt(db_path, num_rows=None):
    # extract create ddls
    '''
    :param root_place:
    :param db_name:
    :return:
    '''
    full_schema_prompt_list = []
    with sqlite3.connect(db_path) as conn:
        # Create a cursor object
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            if table == 'sqlite_sequence':
                continue
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
            create_prompt = cursor.fetchone()[0]
            schemas[table[0]] = create_prompt
            if num_rows:
                cur_table = table[0]
                if cur_table in ['order', 'by', 'group']:
                    cur_table = "`{}`".format(cur_table)

                cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
                column_names = [description[0] for description in cursor.description]
                values = cursor.fetchall()
                rows_prompt = nice_look_table(column_names=column_names, values=values)
                verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(num_rows, cur_table, num_rows, rows_prompt)
                schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for v in schemas.values():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt

def generate_comment_prompt(question, knowledge=None):
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite, answer the following questions for the tables provided above. You can use the provided External Knowledge to help you generate valid and correct SQLite."
    # question_prompt = "-- {}".format(question) + '\n SELECT '
    question_prompt = "-- Question: {}".format(question)

    if knowledge is None:
        result_prompt = pattern_prompt_no_kg + '\n' + question_prompt
    else:
        knowledge_prompt = "-- External Knowledge: {}".format(knowledge)
        result_prompt = knowledge_prompt + '\n' + pattern_prompt_kg + '\n' + question_prompt

    return result_prompt
