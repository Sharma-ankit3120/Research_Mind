from agents import build_search_agent,build_read_agent,writer_chain,critic_chain,llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder


def run_research_pipeline(topic:str)-> dict:

    state = {} # create a local memory




    #Search Agent is working

    print("\n" + "=" * 50)
    print("Step 1 - search agent is working")
    print("="*50)

    search_agent = build_search_agent()
    search_result= search_agent.invoke({
        "messages" : ["user",f"Find recent , reliable and detailed information about: {topic}"]}
    )

    state['search_results']= search_result['messages'][-1].content

    print("\n Search Results: \n", state['search_results'][:800])






    # Read agent is working 

    print("\n" + "=" * 50)
    print("Step 2 - Read agent is scraping top resources")
    print("="*50)

    reader_agent = build_read_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
        })

    state['scraped_content']= reader_result['messages'][-1].content

    print("\n Scraped Content: \n", state['scraped_content'][:800])






    # writer chain is working now

    print("\n" + "=" * 50)
    print("Step 3 - Read agent is scraping top resources")
    print("="*50)
    
    research_combined = (
        f"Search Results : \n {state['search_results']} \n\n"
        f"Detailed Scraped Content: \n {state['scraped_content']}\n"
    )

    state["report"] = writer_chain.invoke({
        "topic" :topic,
        "research" : research_combined
    })

    print("\n final Report: \n", state["report"])






    # critic report

    print("\n" + "=" * 50)
    print("Step 4 - Read agent is scraping top resources")
    print("="*50)
    

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n Critic Report: \n", state["feedback"])


    #### Chat Q/A starts from here

    context = f"""
    TOPIC:{topic}

    ================ FINAL REPORT ==================
    {state["report"]}

    ================ REVIEW ========================
    {state["feedback"]}
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system",
"""
You are an AI Research Assistant.

You have already completed research on a topic.

You are given:

• Search Results
• Scraped Articles
• Final Report
• Critic Review

Your job is to answer the user's questions.

Rules:

1. First use the provided research context.

2. If the answer exists in the context,
answer using it.

3. If some information is missing,
use your own knowledge to complete the explanation.

4. Never hallucinate facts.

5. If something is uncertain,
say it is uncertain.

6. Explain concepts clearly.

7. Maintain conversation history.

Research Context:

{context}
"""
),
    (MessagesPlaceholder(variable_name="chat_history")),
    ("human","{question}")
        ])
    
    

    chat_chain = chat_prompt | llm 

    chat_history = []

    print("\n")
    print("="*60)        
    print("Research Chat Started")
    print("Type 'exit', 'quit', or 'bye' to quit")
    print("="*60)

    while True:

        question = input("\nUser: ")
        if question.lower() in ["exit","quit","bye"]:
            print("\nExiting the research chat. Goodbye!")
            break

        try:
            response = chat_chain.invoke({
                "context": context,
                "chat_history": chat_history,
                "question": question
            })
        
            print("\nAssistant:\n")
            print(response.content)

            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=response.content))

        except Exception as e:
            print(f"\nError: {str(e)}")
            

    return state










    


if __name__ == "__main__":
    topic = input("\n Enter a research topic")
    run_research_pipeline(topic)





