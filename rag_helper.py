INSTRUCTIONS='''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information, 
and make a short and concise answer to the only the given question.

If the answer is not found in the context, say "I don't know". Do not make up an answer.

'''


USER_PROMPT_TEMPLATE='''
Question:
{question}

Context:
{context}
'''

class RAGAbase:
    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model='claude-haiku-4-5-20251001',
        max_tokens=512
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model
        self.max_tokens = max_tokens

    def search(self,query, num_results=5):
        boost_dict={"question":2.0, "section":0.5}
        filter_dict={"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )
    
    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )
    
    def llm(self, prompt):
        response = self.llm_client.messages.create(
            model=self.model,
            system=self.instructions,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=self.max_tokens
        )
        return response.content[0].text.strip()
    
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer