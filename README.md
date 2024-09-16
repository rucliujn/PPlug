# PPlug
LLMs + Persona-Plug = Personalized LLMs
### How to run

- Prepare the Enviroments `bash setup.sh`
- First download the [time-based LaMP dataset](https://lamp-benchmark.github.io/download), [FlanT5-XXL](https://huggingface.co/google/flan-t5-xxl), and [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) here.
- Assemble the dataset (merge the question.json and outputs.josn) `python3.9 aggr_id.py` 
- Compute the historical embedding preivously `python3.9 embedding.py`
- Run the code `cd code && bash run_all_t5.sh`
