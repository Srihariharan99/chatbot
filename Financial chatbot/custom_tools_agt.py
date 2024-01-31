from langchain.llms import GooglePalm
from langchain.utilities import SerpAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain import PromptTemplate
from langchain import  LLMChain
from langchain.agents import load_tools, Tool, initialize_agent,AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
from os import getenv
from dotenv import load_dotenv
load_dotenv()
os.environ["SERPAPI_API_KEY"] = getenv("serpapi_key")

llm_googlepalm = GooglePalm(google_api_key = getenv("googlepalm_key"),temperature = 0.1)

memory = ConversationBufferWindowMemory(
    memory_key = "chat_history",
    return_messages = True
)

search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()
dds = DuckDuckGoSearchRun()
def duck_wrapper (input_text):
  input = (f"site:finance.yahoo.com {input_text}")
  search_results = dds.run(input)
  return search_results




tools = [
     Tool(
        name='DuckDuckGo Search',
        func= duck_wrapper,
        description="Useful for retrieving Current financial data, including stock NEWS, and stock information."
    ),
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input.",
        params={"engine": "google", "google_domain": "google.com", "gl": "us", "hl": "en"}

    ),
    Tool(
        name="Google Finance",
        func=search.run,
        description="Useful for retrieving Current financial data, including stock prices, historical data, and company information.",
        params={"engine": "google_finance", "google_domain": "google.com", "gl": "us", "hl": "en"}
    ),
    Tool(
        name='wikipedia',
        func= wikipedia.run,
        description="Useful for when you need to look up a topic, country or person on wikipedia"
    ),
]

conversation_agent = initialize_agent(tools = tools,
                                      llm = llm_googlepalm,
                                      agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                                      verbose = True,
                                      memory = memory,
                                      early_stopping_method = "generate",
                                      max_iteration = 3,
        
                                      )

def query(prom):
   try:
     a = conversation_agent.run(prom)
     return a
   except:
     print("Kindly ask the appropriate question!")
     



