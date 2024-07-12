import pandas as pd 
from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
import openai
openai.api_key = OPENAI_API_KEY
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import us_engine

happiness_path = 'data/World Happiness Dataset 2015-2023.csv'
df = pd.read_csv(happiness_path)
# print(df.head())
happiness_query_engine = PandasQueryEngine(df=df, verbose=True, instruction_str=instruction_str)
happiness_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=happiness_query_engine,
        metadata=ToolMetadata(
            name="happiness_data",
            description="This provides information about the World Happiness Dataset",
    ),
    ),
    QueryEngineTool(
        query_engine=us_engine,
        metadata=ToolMetadata(
            name="us_data",
            description="This provides detailed information about the United States the country",
    ),
    ),
]

llm= OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
agent = ReActAgent.from_tools(llm=llm, tools=tools, verbose=True, context=context)

while(prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)



# response = happiness_query_engine.query(
#     "What is the least happy country in 2023? and what is the happiness score and its region of the world?",
# )

# print(f'response: {response}')