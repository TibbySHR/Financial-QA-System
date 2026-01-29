# Annual Report QA (A-share) — LLM-Powered Financial Q&A System

## Project Background
Built an interactive Q&A system powered by LLMs to interpret and answer finance-related questions from **10,000+ annual report PDFs** of **A-share listed companies**. Fine-tuned **three domain-specific 7B sub-models** and a **32B summarization model**, using a **small–large model collaboration** architecture to support three fine-grained financial Q&A scenarios: **(1) basic facts lookup**, **(2) statistical/analytical queries**, and **(3) open-ended questions**.

## Key Responsibilities (Finance-focused)
- Implemented a PDF ingestion and extraction pipeline: used **xpdf** to extract **page-level text**, performed **keyword-based page retrieval**, and leveraged **Camelot** for **image-based table extraction**; consolidated and filtered financial statements across **non-consolidated**, **adjusted**, and **parent-company** statements.
- Designed a layered Q&A architecture and created **6K high-quality SFT samples**; fine-tuned a **Qwen2-7B intent classification** model to accurately route user queries to the correct domain/workflow, achieving **94% routing accuracy**.
- Fine-tuned an **NL2SQL** model and a **keyword-extraction** model with **LoRA**: routed **precise** questions to a database-backed NL2SQL engine, and **non-precise** questions to **retrieval + LLM summarization**; built prompt templates by question type and generated answers with **Qwen-32B**, achieving **87% evaluation accuracy**.
- Deployed models with **vLLM** for inference acceleration and containerized the full stack with **Docker**; conducted end-to-end performance/load testing across three sub-models and the full pipeline; enabled **streaming generation** with **0.8s** average **time-to-first-token (TTFT)** and **2.7s** average end-to-end latency.

## Highlights
- Introduced **prefix caching** to speed up inference, reducing **TTFT by 70%**; applied **GPTQ quantization** to **INT8**, reducing average inference latency by **1.1s**; accelerated **32B** inference via **speculative decoding** using **Qwen2-7B** as the draft model. Overall, the system generates answers in **2.7 seconds** on average.
- Enabled a **single-backbone + multi-adapter** deployment using **vLLM + Punica**; leveraged the **SGMV** operator for batched inference across **three LoRA adapters**, cutting GPU deployment resources by approximately **2×**.

## Data Description

### Knowledge Base Composition
- Contains **11,588** annual report PDFs from a subset of listed companies covering **2019–2021**.  
- Total size: **~70 GB**.

### Evaluation Method
- **Task format**: Given a set of **reference documents** and a **question**, the model must generate an answer in a **specified output format**.
- **Test set size**: **1,000** questions.
- Questions cover **multiple types**.

### Example Questions
```json
{"ID": 1, "question": "What was the financial expense of ICBC in 2019 (in CNY)?"}
{"ID": 2, "question": "For ICBC in 2019, what were the non-operating expenses and non-operating income (in CNY), respectively?"}
{"ID": 3, "question": "What was ICBC’s net profit growth rate in 2021? Keep 2 decimal places."}
{
  "ID": 1,
  "question": "What was the financial expense of ICBC in 2019 (in CNY)?",
  "answer": "ICBC’s financial expense in 2019 was 12345678.9 CNY."
}
{
  "ID": 2,
  "question": "For ICBC in 2019, what were the non-operating expenses and non-operating income (in CNY), respectively?",
  "answer": "ICBC’s non-operating expenses in 2019 were 12345678.9 CNY, and non-operating income was 2345678.9 CNY."
}

## Evaluation Metrics

### Evaluation Procedure
An evaluation item is structured as:
- `question`: the user question
- `prompt`: the reference signals used for scoring, including:
  - a target field/value (e.g., `"财务费用": "12345678.9元"`)
  - `key_word`: required keywords that must appear in the model output
  - `prom_answer`: the canonical/target answer string for exact matching
- `answer`: a list of acceptable reference answers (paraphrases)

### Example (Evaluation Sample)
```json
{
  "question": "What was the financial expense of ICBC in 2019 (in CNY)?",
  "prompt": {
    "financial_expense": "12,345,678.9 CNY",
    "key_word": "financial expense, 2019",
    "prom_answer": "12,345,678.9 CNY"
  },
  "answer": [
    "ICBC’s financial expense in 2019 was 12,345,678.9 CNY.",
    "In 2019, ICBC’s financial expense was 12,345,678.9 CNY.",
    "ICBC’s financial expense for 2019 was 12,345,678.9 CNY."
  ]
}

## Project Plan

```mermaid
flowchart LR
  %% ========= Styles =========
  classDef dashed fill:#ffffff,stroke:#3b6ea5,stroke-width:2px,stroke-dasharray:6 4;
  classDef box fill:#4d79c7,stroke:#3b6ea5,stroke-width:2px,color:#ffffff;

  %% ========= Nodes =========
  PDF[pdfAnnual Report]:::box

  TI[Text]:::box
  TBI[Tables]:::box

  QC[Questions Class]:::box

  NL2SQL[NL2SQL]:::box
  KE[Keywords\nExtract]:::box
  RET[Data/text\nRetriveal]:::box
  PROMPT[Prompt\ngenerate]:::box

  GLM[ChatGLM]:::box
  ANS[Answers Genration]:::box
  MET[Metrics Eval]:::box

  %% ========= Dashed Model Blocks =========
  subgraph CLSMODELS[ ]
    direction LR
    CLS_GLM[ChatGLM]:::box
    CLS_PT[Ptuing-v2]:::box
  end
  class CLSMODELS dashed;

  subgraph NL2SQLMODELS[ ]
    direction LR
    N2S_GLM[ChatGLM]:::box
    N2S_PT[Ptuing-v2]:::box
  end
  class NL2SQLMODELS dashed;

  %% ========= Flow =========
  PDF --> TI
  PDF --> TBI

  TI --> QC
  TBI --> QC

  CLSMODELS --> QC

  QC --> NL2SQL
  QC --> KE

  NL2SQLMODELS --> NL2SQL

  KE --> RET --> PROMPT --> GLM --> ANS --> MET

  %% NL2SQL branch goes directly to ChatGLM (as in your figure)
  NL2SQL --> GLM

  %% ========= Red Links (approx) =========
  linkStyle 0,1,2,3,5,6,7,8,9,10,11 stroke:#d73a49,stroke-width:2px;
  linkStyle 4 stroke:#d73a49,stroke-width:2px;       %% CLSMODELS -> QC
  linkStyle 12 stroke:#d73a49,stroke-width:2px;      %% NL2SQLMODELS -> NL2SQL
  linkStyle 13 stroke:#d73a49,stroke-width:2px;      %% NL2SQL -> GLM


## Code Structure

## Repository Structure

```text
.
├── README.md
├── __init__.py
├── chatglm_ptuning.py
├── check.py
├── company_table.py
├── config/
├── data/
├── file.py
├── financial_state.py
├── generate_answer_with_classify.py
├── image/
├── main.py
├── pdf_util.py
├── preprocess.py
├── prompt_util.py
├── ptuning/
├── question_util.py
├── re_util.py
├── recall_report_names.py
├── recall_report_text.py
├── requirements.txt
├── sql_correct_util.py
├── test_score.py
├── type1.py
├── type2.py
├── version.py
└── xpdf/

![Extraction](./src/image/text-extract.png)
## 8. Workflow — Question Classification

To generate accurate answers with an LLM, the prerequisite is understanding the user’s question and intent. For different question categories, the system applies different processing strategies.

We can fine-tune a classification model to automatically categorize user questions. In practice, this means building a “classifier” with the following steps:

1. First, generate a dataset using few-shot **in-context learning**, then manually validate/clean it.
2. Next, fine-tune the model (e.g., **LoRA** or **P-Tuning**) on this dataset to train a question classification model.
3. Finally, feed user questions directly into the model and let it select and output the corresponding category.

Question classification typically maps text queries into predefined categories. We build the classification prompt template with the following steps:

- **Define categories**: First, clearly define the categories in the task. These categories are designed based on the types of questions in the dataset.

- **Strengthen option descriptions**: Enhance the model’s understanding of each option by adding category descriptions and category-related keywords. Below is the constructed classification prompt:

```python
def _get_classify_prompt(self, question) -> str:
    classify_prompt = """
Please determine which category the following question belongs to:

A: Company basic information, including stock abbreviation, company name, foreign name, legal representative, registered address, office address, company website, email, etc.

B: Company employee information, including number of employees, employee specialties, employee types, education level, etc.

C: Financial-statement-related content, including fields appearing in the balance sheet, cash flow statement, and income statement, such as expenses, assets, amounts, revenues, etc.

D: Computation questions that cannot be directly obtained from the annual report and must be derived using formulas, including growth rates, ratios, proportions, shares, etc.

E: Statistical questions that require extracting search conditions from the question and then retrieving/filtering/sorting in the dataset/database to obtain the result.

F: Open-ended questions, including introductions, methods, analysis, impact assessment, and “What is XXX?”-type explanatory questions.

You only need to output the letter label. Do not output anything other than the letter label.
""".format(question)

    return classify_prompt

### Questions classfication

```mermaid
flowchart LR
  %% ===== Styles =====
  classDef dash fill:#ffffff,stroke:#2459d3,stroke-width:2px,stroke-dasharray:7 5;
  classDef hdr fill:#7a2dd8,stroke:#5b22a3,stroke-width:2px,color:#ffffff;
  classDef box fill:#ffffff,stroke:#2459d3,stroke-width:2px,stroke-dasharray:7 5;
  classDef arrow stroke:#6b2bd8,stroke-width:3px;

  %% ===== Columns (3 panels) =====
  subgraph R[Rule-based]
    direction TB
    R1["✓ Contains company name / year"]:::box
    R2["✓ Matches statement/table fields"]:::box
    R3["✓ Matches formulas"]:::box
    R4["✓ Special terminology"]:::box
  end
  class R dash;

  subgraph P[LLM Prompt]
    direction TB
    H["LLM Prompt"]:::hdr

    T["Determine which category the question belongs to:\n\nA: Basic company info (e.g., ticker, stock code, company name, legal representative, registered address, office address, website, email)\n\nB: Employee statistics (e.g., headcount, roles, education level)\n\nC: Finance-related fields (e.g., amount, expense, assets, revenue)\n\nD: None of the above / open-ended analysis\n\nExamples:\n1) What is XXXX's expense/revenue? → Output: C\n2) Who is XX company's legal representative? → Output: A\n3) Briefly analyze XX company's XXX situation. → Output: D\n4) How many master's degree employees does XX company have? → Output: B\n\nOnly output the letter."]:::box
  end
  class P dash;

  subgraph PT[P-Tuning]
    direction TB
    PT1["PRE_SEQ_LEN = 512\nLR = 2e-2\nmax_source_length = 512\nmax_target_length = 128"]:::box
  end
  class PT dash;

  %% ===== Arrows =====
  R --> P
  P --> PT

  %% color arrows (approx)
  linkStyle 0,1 stroke:#6b2bd8,stroke-width:4px;


## Key Code

### Model Fine-tuning
- `./ptuning/CLASSIFY_PTUNING/train.sh`

### Online Inference
- `main.py`: Lines **86–88**
- `generate_answer_with_classify.py`: `def do_classification()`

---

## 9. Workflow — SQL Generation (NL2SQL)

The key focus of NL2SQL fine-tuning is **building a high-quality training dataset**. It must be emphasized that the dataset should be constructed for **real-world financial query scenarios**, rather than relying on generic benchmarks such as **Spider**. Two common approaches are:

1. **Template-based SQL data generation**  
   - Pros: simple and fast to scale  
   - Cons: relatively **weak generalization**

2. **Build SQL Q&A templates + randomly fill fields + LLM paraphrasing**  
   - Method: design SQL and corresponding Q&A templates, randomly sample/fill slots (fields/conditions), then use ChatGPT or other LLMs to rewrite/paraphrase questions to increase diversity  
   - Result: typically **better performance** and closer to real user queries

Regardless of the approach, the dataset must be **manually validated**. For fine-tuning, **data quality is far more important than data quantity**.

The core problem becomes: **how to generate correct SQL queries to retrieve accurate data**.


### Three Recommended Practices (Summary)

1. **Use a single wide table (denormalized schema).**  
   With a single table, SQL for single-table queries is much simpler, which significantly improves NL2SQL accuracy.

2. **Explicitly specify target columns in the prompt.**  
   Clearly constraining the model to specific fields/columns improves correctness. To accurately locate the right table columns, you can use the keyword-matching approach described earlier.

3. **Decompose computation-heavy questions into multiple steps.**  
   For scenarios that require calculations (e.g., **Operating Margin = Operating Profit / Operating Revenue**), it is better to split the problem into multiple simple SQL queries, retrieve the required values separately (operating profit, operating revenue), and then compute the final metric using the formula.

---

## NL2SQL Template Construction

NL2SQL converts natural-language questions into SQL queries to retrieve information from a database. Below are the steps for building an NL2SQL prompt/template:

- **Define query types**: retrieval, sorting, output range constraints, counting, summation, single-field search, multi-field search, multi-condition multi-field search, field filtering, and other common SQL execution patterns.

- **Example dialogues**: guide the LLM (as an agent acting like a MySQL developer) using paired examples of natural-language questions and corresponding SQL queries, so the model learns the intended instruction-following behavior.

- **Database fields**: provide the database table name and the designed columns. The database should contain well-cleaned and merged data; searching within the same type of fields should be performant.

- **Generate SQL**: generate correct executable SQL statements following a SQL syntax tree / structured SQL grammar.

- **Parameter substitution**: replace parameters in the natural-language question (e.g., column names, conditions) with valid database elements that actually exist in the schema.

## NL2SQL & Keyword Extraction — What This Section Means (English)

### NL2SQL: Fine-tuning & Online Inference

- **Model fine-tuning**
  - `./ptuning/NL2SQL_PTUNING/train.sh`
  - Meaning: this is the training entry point used to fine-tune the **NL2SQL** model with **P-Tuning** (so it can generate SQL that matches your schema and query patterns).

- **Online inference (where NL2SQL is called at runtime)**
  - `main.py`: Lines **96–98**
  - `generate_answer_with_classify.py`: `def do_sql_generation()`
  - Meaning: these are the runtime integration points where the system invokes the NL2SQL component to produce SQL for “precise” questions.

---

## 10. Workflow — Keyword Extraction

In many scenarios, the system uses a **prompt** to ask an LLM to extract keywords. The purpose is to extract the most important terms from the user question so the system can:
1) better understand intent, and  
2) route the question to the correct answering strategy (rules, NL2SQL, retrieval + summarization, etc.).

### Why a keyword model is needed
For domain-specific finance questions, a general model (e.g., ChatGLM2-6B) may not reliably understand or extract specialized terminology. In that case, the project fine-tunes a dedicated **keyword-extraction model** to extract professional financial keywords from user queries more accurately.

### How it is used in practice
A common pipeline is:
1. Use the keyword model to **summarize** the user question and **extract key information** (keywords/slots).
2. **Vectorize** the extracted keywords (embeddings).
3. Perform **vector search** to match relevant keywords (or retrieve relevant content).
4. Use the matched keywords to **locate the corresponding database fields** more precisely.

### Why it matters
Keyword extraction focuses on extracting the most representative words/phrases from text, which improves downstream retrieval quality. In general, **the more accurate the extracted keywords, the more accurate the retrieved evidence**, and therefore the final answer.

### Building the keyword-extraction prompt
- Construct a keyword-extraction prompt and continuously test its stability on diverse questions.
- Even without fine-tuning, **few-shot prompting** can achieve acceptable performance.
- The section then provides the project’s prompt template starting from:
  - `role_prompt = ''' ...`

## 10. Workflow — Keyword Extraction

In many cases, we use **prompting** to ask a large language model (LLM) to **extract keywords**.

For example, by extracting keywords from a user’s question, we can better understand the user’s intent and then apply rule-based matching to decide how to answer the question.

However, in specialized domains (finance), a general model such as **ChatGLM2-6B** may not reliably understand the domain context or accurately extract professional terms. In that situation, we can **fine-tune a dedicated keyword-extraction model** to extract domain-relevant keywords from user queries.

In real usage, we can first use the keyword model to **summarize the user query and extract key information**, then **vectorize** that information and run **vector search** to match the most relevant keywords. This approach can also help us **precisely locate the corresponding fields/columns in the database schema**.

Keyword extraction aims to extract key words or short phrases from text, which helps downstream retrieval of relevant information. In general, **more accurate keyword extraction leads to more accurate retrieved evidence**.

### Steps to build a keyword-extraction template

- **Build the keyword-extraction prompt**  
  Continuously test the prompt for robustness and stability across different question types. Even without fine-tuning, a **few-shot** setup can often achieve acceptable performance. Below is the prompt format used in this project:

```python
role_prompt = '''
Please extract keywords from the sentence below. These keywords should be the most important words that best summarize the main topic of the sentence. By using these keywords, you can better understand the sentence. Only output the keywords from the text. Do not output anything else.

User input:
'''
question_prompt = role_prompt + question


## Sample Output

**Question:** Please briefly describe the delisting situation faced by Chengdu Galaxy Magnets Co., Ltd. in 2019.  
**Keywords:** potential delisting situation

**Question:** How much accounts receivable financing did Hunan Jiudian Pharmaceutical Co., Ltd. have in 2020?  
**Keywords:** accounts receivable financing

**Question:** What was Xiamen Tungsten’s current ratio in 2020?  
**Keywords:** current ratio

**Question:** In 2020, what were the investment income and net profit of Guangxi Boshike Environmental Technology Co., Ltd., respectively?  
**Keywords:** "investment income", "net profit"

### Key Code

**Model fine-tuning:**  
`./ptuning/KEYWORDS_PTUNING/train.sh`

**Online inference:**  
- `main.py`: Line 91–93  
- `generate_answer_with_classify.py`: `def do_gen_keywords()`###

### Prompt Design

flowchart TB
    P((Prompt))

    A["1) Role Definition<br/><br/>Be precise; match the background knowledge required to complete the task."]
    B["2) Task Goal Description<br/><br/>Keep it concise and clear; avoid vague or ambiguous wording."]
    C["3) Specific Requirements, Reasoning, Steps<br/><br/>Use bullet points; avoid negative instructions.<br/>If the workflow is long, split it into multiple sub-tasks."]
    D["4) Examples<br/><br/>Representative examples that cover different types/cases."]
    E["5) Output Requirements<br/><br/>Explicitly specify the output format; re-emphasize critical constraints."]

    A --- P
    B --- P
    C --- P
    D --- P
    E --- P


