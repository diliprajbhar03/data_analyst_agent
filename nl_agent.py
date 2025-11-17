# nl_agent.py
import os
import openai
import json
import pandas as pd
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = '''You are a helpful Data Analyst Assistant that translates short natural-language user requests into a safe JSON action describing what to do with a pandas DataFrame.

Input:
- `instruction`: the user's instruction (single sentence or short paragraph)
- `columns`: a list/dict of column names and types

Output (strict JSON):
{
  "action": "one of [select, aggregate, filter, describe, plot, train, forecast]",
  "details": { ... }  // action-specific details
}

Examples:
- "show average sales per month for region east" -> {"action":"aggregate","details":{"groupby":"month","agg":"mean","column":"sales","filter":{"column":"region","op":"=","value":"east"}}}
- "plot revenue vs marketing_spend" -> {"action":"plot","details":{"kind":"scatter","x":"marketing_spend","y":"revenue"}}

Respond ONLY with JSON.
'''

def nl_to_action(instruction, columns_sample):
    prompt = PROMPT_TEMPLATE + "\n\nInstruction:\n" + instruction + "\n\nColumns:\n" + json.dumps(columns_sample) + "\n\nRespond with JSON:"
    resp = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], temperature=0)
    txt = resp['choices'][0]['message']['content']
    try:
        j = json.loads(txt)
        return j
    except Exception:
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            return json.loads(m.group(0))
        raise ValueError("Could not parse LLM output as JSON")

def execute_action(df, action_json):
    act = action_json.get('action')
    d = action_json.get('details', {})
    if act == 'describe':
        return df.describe(include='all')
    if act == 'select':
        cols = d.get('columns')
        return df[cols]
    if act == 'filter':
        col = d['column']; op = d.get('op','=='); val = d['value']
        if op in ['==','=', 'eq']:
            return df[df[col] == val]
        if op in ['!=','ne']:
            return df[df[col] != val]
        if op == '>':
            return df[df[col] > val]
        if op == '<':
            return df[df[col] < val]
    if act == 'aggregate':
        groupby = d.get('groupby'); agg = d.get('agg','mean'); col = d.get('column')
        if groupby == 'month':
            df2 = df.copy(); df2['_month'] = pd.to_datetime(df2[d.get('date_col')]).dt.to_period('M')
            return df2.groupby('_month')[col].agg(agg).reset_index()
        else:
            return df.groupby(groupby)[col].agg(agg).reset_index()
    if act == 'plot':
        return {'plot': d}
    raise ValueError('Unsupported action for execution')
