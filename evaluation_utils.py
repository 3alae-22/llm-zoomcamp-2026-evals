import time

from tqdm.auto import tqdm
from rag_helper import RAGAbase

model_base = "claude-haiku-4-5-20251001"

USD_TO_MAD = 10.0  # 1 USD ≈ 10 MAD (à ajuster selon le taux du jour)

def calc_price(usage):
    input_price_per_million       = 0.25
    output_price_per_million      = 1.25
    cache_write_price_per_million = 0.30
    cache_read_price_per_million  = 0.03

    input_cost  = (usage.input_tokens  / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million

    cache_write_tokens = (usage.cache_creation_input_tokens or 0)
    if hasattr(usage, 'cache_creation') and usage.cache_creation:
        cache_write_tokens += (usage.cache_creation.ephemeral_1h_input_tokens or 0)
        cache_write_tokens += (usage.cache_creation.ephemeral_5m_input_tokens or 0)
    cache_write_cost = (cache_write_tokens / 1_000_000) * cache_write_price_per_million

    cache_read_tokens = (usage.cache_read_input_tokens or 0)
    cache_read_cost   = (cache_read_tokens / 1_000_000) * cache_read_price_per_million

    total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost

    return {
        "input_cost_usd":       input_cost,
        "output_cost_usd":      output_cost,
        "cache_write_cost_usd": cache_write_cost,
        "cache_read_cost_usd":  cache_read_cost,
        "total_cost_usd":       total_cost,
        # En DHS
        "input_cost_mad":       input_cost       * USD_TO_MAD,
        "output_cost_mad":      output_cost      * USD_TO_MAD,
        "cache_write_cost_mad": cache_write_cost * USD_TO_MAD,
        "cache_read_cost_mad":  cache_read_cost  * USD_TO_MAD,
        "total_cost_mad":       total_cost       * USD_TO_MAD,
    }


def calc_total_price(usages):
    total_usd = 0.0
    for usage in usages:
        cost = calc_price(usage)
        total_usd += cost["total_cost_usd"]
    return {
        "total_usd": total_usd,
        "total_mad": total_usd * USD_TO_MAD,
    }

def llm_structured(client, instructions, user_prompt, output_type, model=model_base):
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    response, completion = client.messages.create_with_completion(
        model=model,
        system=instructions,
        messages=messages,
        response_model=output_type,
        max_tokens=512
    )
    return response, completion.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model=model_base,
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGAbase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.messages.create(
            model=self.model,
            system=self.instructions,
            messages=input_messages,
            max_tokens=512
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.content[0].text

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results
