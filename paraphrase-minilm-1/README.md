---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:637
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/paraphrase-MiniLM-L6-v2
widget:
- source_sentence: stream used managing large bulk data like image binary data memory
    efficiently stream 4 type readable writable duplex transform data stream manage
    large amount data chopping small chunk data itll ensure memory efficiency reduce
    load system using stream need use extra memory function "fs" module are fscreatereadstream
    itll read data chopping large amount data simple unit ensure memory efficiency
    fscreatewritestream itll write data chopping large amount data simple unit ensure
    load efficiency memory efficiency better use fscreatereadstream using fsreadfile
  sentences:
  - const fsrequirefs function readlogfilefilepath fsreadfilefilepathutf8errdata iferr
    return err else return data consolelogreadlogfileinputtxt
  - stream type reading writing function nodejs either reading writing done part known
    chunk chunk part data sent read part part save data make consistent transfer data
    type transfer known stream used data secure confidential sent large group audiance
    time use method createreadstream createwritestream method reading writing file
    data form stream
  - node single thread working model single thread handle operationsso also called
    single thread machine even though node j single thread non blocking even get excuted
    get event go event queue process go event loop handle even get excuted one one
    thread make process efficent make non blocking first get event go event queue
    go event loop concurrent way handled thread large data prefered node j like video
    etc single thread single threadconcurrency event loop work
- source_sentence: expressjs web framework used developing fast scalable web application
    easily effortlessly express perform nonblocking io operation efficiently using
    express complex task like routing done effortlessly express used building fast
    scalable web application implementing rest apis easier using express
  sentences:
  - rest api stateless api used create url endpoint allows two software communicate
    other transfer data using rest apis rest apis data format json eg sample code
    rest api using get request appuseexpressjson appgetdatareq re try functionality
    resstatus200jsondata catcherror resstatus500sendinternal server error
  - read logfile mainly used put timestrams code function readlogfilefilepath fsreadfilefilepathutf8reserr
    iferr consolelogerr return re code creating readfile read file content particular
    file using asynchronous readfile methos read file content using call back handle
    promise
  - express expressjs nodejs framework expressjs make easier work route also make
    easier response request send user also us mvc architecture design pattern nodejs
    use express importing module express need install express npm express example
    const express requireexpressimporting express const app expresscreating instance
    appgetusersroutes
- source_sentence: nodejs nodejs single threaded language perform one task time order
    manage multiple operation use asynchronus method multiple operation run concurrently
    according time take execute handle asynchronus programming use async await keywords
    using keywords make asynchronus method look like synchronus way handle concurrency
    execution operation nodejs example let u consider file sampletxt containing text
    hello gradious const f requirefspromise synchronus method const myfun filepath
    const data fsreadfilefilepathutf8 consolelogdata output data promisepending asynchronus
    method const myfunction async filepath const data await fsreadfilefilepathutf8
    consolelog"data "+data output data hello gradious myfunctionsampletxt myfunsampletxt
    executing two method functionality get different output
  sentences:
  - first create server const createserverrequirenodehttp const hostname127001 const
    port8080 const servercreateserverreqres resstatus200 ressetheadercontenttype resendhello
    world serverlistenporthostname consolelogserver listening httphostnameport output
    server listening http1270018080 http request performing curd operation like get
    put delete post get retrives us putupdates user data deletedelete user data postappend
    new user example const expressrequireexpress const appexpress appget consoleloghome
    page applisten8080 cosolelogserve listening httplocalhost8080 basic node express
    application
  - middleware crossing edge act interface request response in middleware verifying
    input output ie request response server with help middle ware authenticate authorize
    teh use info make crucial day withe help middle ware vefrify input also with help
    middle mulple input out block sends input sequential order help single thread
  - nodejs single thread nodejs single thread execution known synchronous internally
    constists event loop callback function due blocking execution code event organized
    event queue according executed nodejs concurrency nodejs externally appears e
    synchronous internally different asynchronous execution performed thus execution
    concurrent perfomed according loop event concurrently nodejs eventloop nodejs
    event executed single thread according event loop event loop consists several
    event organized event queue event executed one queue also consists callback
- source_sentence: const express requireexpress const app express appuse req re ressend"hello
    world" const port 8080 applistenport consolelogserver running httplocalhostport
  sentences:
  - expressjs used basically read stream data either local file external database
    enhances application data integrity enhances user experience also used create
    file stream large amount data broken chunk streamed onto client application reduces
    application memory us less system resource compared normal fsreadfilesync
  - const express requireexpress const app express const port 8080 appgetreqres ressendhello
    world applistenport consolelogrunning port 8080
  - moduleloggerjs class logger static info const re "this information message" return
    re static warning const re "this warning message" return re static error const
    re "this error message" return re moduleexports logger routersloggerrouterjs const
    logger requiremodulelogger const router expressrouter routergetinfo loggerinfo
    routergetwarning loggerwarning routergeterror loggererror moduleexports router
    serverjs const loggerrouter requireroutersloggerrouter const logger requirelogger
    appuseapi loggerrouter appget req re ressend"this main route" const port 5000
    applistenport consolelog"server running port" port
- source_sentence: const f requirefs const path requirepath function readlogfilefilepath
    const file pathjoindirname filepath fsreadfilefile utf8 err re iferr return err
    return re const content readlogfilelibdatatxt consolelogcontent
  sentences:
  - const f require"fs" const f promisifyfs function readlogfilefilename const data
    fsreadfilefilename "utf8" return data consolelogreadlogfile"dumpslog"
  - single thread nodejs handle data asynchronously also nodejs single threaded handle
    multiple reqests give respones take place asynchronus operation concurrency handling
    multiple request response time non blocking operation wait one process complete
    exectue another process executes process simentaneouly event loop handle synchronous
    an asunchronous operation send synchronous operation thread pool initially thread
    pool contains 4 thread asynchronous handle directly complete execution response
    sent server sent client
  - http header sent along http request contains detail type data sent also send token
    authorization purpose may contain information data send send http header using
    axios module fetch request 1 axios const axios requireaxios const header contenttype
    applicationjson authorization jsonstringifytoken axiosgethttplocalhost8080headers
    2 fetch fetchhttplocalhost8080 method get header contenttype applicationjson authorization
    jsonstringifytoken following way send http request server
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/paraphrase-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) <!-- at revision 9a27583f9c2cc7c03a95c08c5f087318109e2613 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'const f requirefs const path requirepath function readlogfilefilepath const file pathjoindirname filepath fsreadfilefile utf8 err re iferr return err return re const content readlogfilelibdatatxt consolelogcontent',
    'const f require"fs" const f promisifyfs function readlogfilefilename const data fsreadfilefilename "utf8" return data consolelogreadlogfile"dumpslog"',
    'http header sent along http request contains detail type data sent also send token authorization purpose may contain information data send send http header using axios module fetch request 1 axios const axios requireaxios const header contenttype applicationjson authorization jsonstringifytoken axiosgethttplocalhost8080headers 2 fetch fetchhttplocalhost8080 method get header contenttype applicationjson authorization jsonstringifytoken following way send http request server',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 637 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 637 samples:
  |         | sentence_0                                                                          | sentence_1                                                                         | label                                                           |
  |:--------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------|
  | type    | string                                                                              | string                                                                             | float                                                           |
  | details | <ul><li>min: 42 tokens</li><li>mean: 86.85 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 2 tokens</li><li>mean: 75.76 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 6.24</li><li>max: 10.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | sentence_1                                                                                                                                                                                                                                                                                  | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>nodejs nodejs single threaded language perform one task time order manage multiple operation use asynchronus method multiple operation run concurrently according time take execute handle asynchronus programming use async await keywords using keywords make asynchronus method look like synchronus way handle concurrency execution operation nodejs example let u consider file sampletxt containing text hello gradious const f requirefspromise synchronus method const myfun filepath const data fsreadfilefilepathutf8 consolelogdata output data promisepending asynchronus method const myfunction async filepath const data await fsreadfilefilepathutf8 consolelog"data "+data output data hello gradious myfunctionsampletxt myfunsampletxt executing two method functionality get different output</code> | <code>nodejs single thread refers list nonasynchronous operation follows order execution situation concurrency maintained concurrency hand refers property list asynchronous operation flexibility execute time without disturbing flow execution event loop one handle task one one</code> | <code>5.0</code> |
  | <code>header http used security purpose ie help encrypting data sent url header different type applicationjson textplain set header using function setheader"contenttype" "applicationjson"</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>header http request used mention header passed particular http request header sent json key value pair main key header example headerscontenttypeapplicationjson header may also include authentcation cooky token content</code>                                                     | <code>4.0</code> |
  | <code>stream used managing large bulk data like image binary data memory efficiently stream 4 type readable writable duplex transform data stream manage large amount data chopping small chunk data itll ensure memory efficiency reduce load system using stream need use extra memory function "fs" module are fscreatereadstream itll read data chopping large amount data simple unit ensure memory efficiency fscreatewritestream itll write data chopping large amount data simple unit ensure load efficiency memory efficiency better use fscreatereadstream using fsreadfile</code>                                                                                                                                                                                                                                   | <code>whenever large amount data need transferred stream used stream divide large data smaller easy transfer part called chunk sends data form smaller sized chunk reduce amount memory required run application increase compile time</code>                                               | <code>4.0</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 2
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.1.1
- Transformers: 4.45.2
- PyTorch: 2.5.1+cu121
- Accelerate: 1.1.1
- Datasets: 3.1.0
- Tokenizers: 0.20.3

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->