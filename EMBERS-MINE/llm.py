import openai
import tiktoken
import json
from config import Config

class LLM():
    ###
    # LLM関連の処理を実行するクラス
    ###
    def __init__(self, 
                 api_key='', 
                 model_name='gpt-3.5-turbo'):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
    
    def compute_num_token(self,
                          text=''):
        return len(self.tokenizer.encode(text))

    def truncate(self, input_text, max_tokens):
        truncated_text = self.tokenizer.decode(
            self.tokenizer.encode(input_text)[:max_tokens]
        )
        return truncated_text
    
    def openai_wrapper(self,
                       system_setting_prompt='',
                       user_input=''):
        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_setting_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.1,
                response_format={ "type": "json_object" },
                seed=8888,
                n=1,
                stop=None,
        )
        try:
            result_json = response.choices[0].message.content.strip()
            if result_json.startswith('```json'):
                result_json = result_json.replace('```json', '')
                result_json = result_json.replace('```', '')
        except Exception as e:
            print(e)
            result_json = None
        return result_json

    def generate_long_output(self,
                             system_setting_prompt='',
                             user_input=''):
        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_setting_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.1,
                seed=8888,
                n=1,
                stop=None,
        )
        output = response.choices[0].message.content.strip()
        if response.choices[0].finish_reason == 'length':
            print('*****Output*****', output)
            # 出力が最大出力トークン数を超えた場合
            updated_user_input = user_input + '\n\nYour output:\n' + output + '\nPlease go on with the rest of the JSON and complete the JSON output.'
            print('*****Updated User Input*****', updated_user_input)
            return self.generate_long_output(system_setting_prompt=system_setting_prompt,
                                        user_input=updated_user_input)
        else:
            try:
                result_json = response.choices[0].message.content.strip()
                if result_json.startswith('```json'):
                        result_json = result_json.replace('```json', '')
                        result_json = result_json.replace('```', '')
            except Exception as e:
                print(e)
                result_json = None
            return result_json

    
    def determine_target_study_or_not(self,
                                      abstract_text='',
                                      method_text=''):
        system_setting_prompt = '''
Analyze the abstract and methods section to determine if the paper describes an original research study that includes metagenomic or 16S rRNA gene amplicon analysis of newly collected human fecal samples. The study should not be a review, meta-analysis, tool development, or drug testing on cultured cells. Studies focusing on non-human subjects such as mice, rats, or primates should also be excluded. The study may include analysis of other human body sites (e.g., oral microbiome) or other omics data (e.g., metatranscriptomics, metabolomics), but should not be purely focused on non-gut microbiome or other omics.

Confirm if the study likely involved analyzing the human gut microbiome using original human fecal samples based on the information provided in the methods section, even if the abstract does not explicitly mention the specific analysis methods (16S or shotgun).

After analyzing the text with the prompts, structure your JSON output based on the findings. Here's an example of how to format the output:
{
  "decision": "yes/no",
  "reason": "The abstract and methods section [suggest/do not suggest] that the study is an original research involving metagenomic or 16S rRNA gene amplicon analysis of newly collected human fecal samples. [Although/And] it may also involve analysis of other human body sites or other omics data, the focus [appears to/does not appear to] include the human gut microbiome. [Add specific phrases or sentences from the text that led to this conclusion, if available.]"
}

Replace "yes/no" with the decision based on the analysis and provide a rationale that includes evidence from the text, if present. If the methods section provides clear evidence that the study includes original human gut microbiome analysis, the decision should be "yes" even if the abstract does not provide conclusive information.
'''

        user_input = f'''
Abstract text:
{abstract_text}

Methods text:
{method_text}
'''
        if self.compute_num_token(user_input) > Config.MAX_TOKENS:
            # truncate input text
            user_input = self.truncate(user_input, Config.MAX_TOKENS)

        return self.openai_wrapper(system_setting_prompt=system_setting_prompt,
                                   user_input=user_input)

    def analyze_project_info(self,
                             schema='',
                             abstract_text='',
                             method_text=''):
        system_setting_prompt = '''
Analyze the provided text from the abstract and materials-and-methods sections of a human gut microbiome research paper to extract the following information:

Analysis Method: Determine whether the study uses shotgun metagenomic analysis, 16S rRNA gene amplicon analysis, or both methods. Look for specific mentions of these methodologies. If found, use one of these values: "shotgun", "amplicon", "both". 
Subject Type: Identify if the study subjects were adults or infants, based on the context or explicit mentions. If found, use one of these values: "adult", "infant", "both", "other".
Country: Identify the country or countries in which the subjects reside or where the sampling was performed. This may be explicitly stated or inferred from the institution's location. If a country is in the provided list, use the country name exactly as it appears. If a country is not in the list, include the mentioned country name as is. Output the country names as a list.
Key Findings: Summarize the main results of the study in one sentence, focusing on the most significant outcomes. 
Disease (Optional): If the study specifically mentions conducting research on patients with a disease or diseases, note the name(s) of the disease(s). If found, use the disease name(s) exactly as it appears in the provided disease list. If a disease is not in the list, include the mentioned disease name as is. Output the disease names as a list.

For elements that cannot be determined from the given text, provide an empty list ([]) for that element.

Organize the extracted information in JSON format, using the specified categories as keys. 

Here's an output format:
{
  "analysis_method": "shotgun/amplicon/both",
  "subject_type": "adult/infant/both/other", 
  "country": [],
  "key_findings": "",
  "disease": [],
}

If certain information isn't available or cannot be determined from the provided text, use an empty list ([]) for "country" and "disease", and an empty string ("") for other elements.
'''

        user_input = f'''
Schema:
{schema}
        
Abstract:
{abstract_text}

Materials and Methods:
{method_text}
'''
        
        if self.compute_num_token(user_input) > Config.MAX_TOKENS:
            # truncate input text
            user_input = self.truncate(user_input, Config.MAX_TOKENS)

        return self.openai_wrapper(system_setting_prompt=system_setting_prompt,
                                   user_input=user_input)

    def analyze_methods(self,
                        method_text=''):
        system_setting_prompt = '''
You are a program designed to analyze the Methods section of microbiome research papers.
Your task is to extract detailed information about sample collection and preservation, DNA extraction methods, and 16S rRNA amplicon analysis specifics, including the amplified regions and primer sequences if mentioned.
Your output should be formatted as a JSON object with three main sections: 'Sampling', 'DNA extraction', and 'Amplicon', each containing the relevant extracted information.
If the paper does not provide specific details on any of these sections, leave the corresponding field empty or as an empty list.
Note that if the paper discusses shotgun metagenomic analysis instead of 16S rRNA amplicon analysis, the 'Amplicon' section should be omitted or filled with empty strings for its sub-fields.

For the 'DNA extraction' section, break down the method into detailed steps, providing a list of sentences that corresponds to each experimental step (including any kits, enzymes, or other product names) mentioned.
Ensure each step is expressed as a standalone, clear sentence according to the following specific guidelines:
1. Isolate the description of the DNA extraction process and list each step involved in this process sequentially.
2. Revise or rephrase each step to form a coherent sentence that does not require context from the previous sentences.
3. Exclude any transitional phrases such as 'Finally', 'Then', 'Afterwards', etc., making sure each step is independent and self-contained.
4. Combine any actions described across multiple sentences into a single, concise sentence for each experimental step.
5. Do not include steps related to DNA quality assessment, quantification, or library preparation for sequencing.
6. Avoid including any descriptions of experimental results or outcomes in each step.

Before you start analyzing the provided text, here are an examples to guide your extraction process:

Input Text:
"Fecal samples were collected using sterile techniques and immediately frozen at -80°C for later DNA extraction. Metagenomic DNA was isolated from the fecal samples using ZR Fecal DNA MiniPrepTM. Up to 150 mg of each frozen fecal sample was scraped off with the help of a sterilized spatula (to avoid freeze thaw and microbial contamination) and transferred into a tube provided with the kit. After addition of lysis buffer, samples were homogenized by FastPrep 120 at speed 6 m/s for 40 seconds followed by cooling on ice for 1 minute, thrice. The rest of the protocol was performed as per the instructions in the user manual. This kit does not contain any RNAse step, so the nucleic acids isolated by the ZR kit were treated with 10 μl of RNAse (10 mg/ml stock) for 30 minutes at 37 ̊C to remove any contaminating RNA. Following this, the sample was subjected to phenol/Sevag extraction and the nucleic acids recovered from the aqueous phase by ethanol precipitation and suspended in TE. The V1–V2 regions of 16S rRNA gene were selected for pyrosequencing analysis because that region offers enough sequence variation for species-level discrimination of bifidobacteria, which are reported as the main constituent of infant gut microbiota. The 16S rRNA genes of each sample were amplified using a forward 63Fm-TAG-linker A primer and a reverse 338R-linker B primer."

Output:
{"Sampling": "Fecal samples were collected using sterile techniques and immediately frozen at -80°C for later DNA extraction.",
 "DNA extraction":
        ["Metagenomic DNA was isolated from the fecal samples using ZR Fecal DNA MiniPrepTM."
         "Up to 150 mg of each frozen fecal sample was scraped off with the help of a sterilized spatula and transferred into a tube provided with the kit."
         "After addition of lysis buffer, samples were homogenized by FastPrep 120 at speed 6 m/s for 40 seconds and then cooled on ice for 1 minute, repeated thrice."
         "The nucleic acids isolated by the ZR kit were treated with 10 μl of RNAse (10 mg/ml stock) for 30 minutes at 37°C to remove any contaminating RNA."
        "The sample was subjected to phenol/Sevag extraction and the nucleic acids were recovered from the aqueous phase by ethanol precipitation and suspended in TE."]
 "Amplicon": {"Amplified regions": "V1-V2",
              "Forward primer": "63Fm-TAG-linker A",
              "Reverse primer": "338R-linker B"}

Based on this example, please analyze the following text from the Methods section of a microbiome research paper and extract the relevant information into the specified JSON format."
'''

        user_input = f'''
Materials and Methods:
{method_text}
'''
        if self.compute_num_token(user_input) > Config.MAX_TOKENS:
            # truncate input text
            user_input = self.truncate(user_input, Config.MAX_TOKENS)

        return self.openai_wrapper(system_setting_prompt=system_setting_prompt,
                                   user_input=user_input)


    def judge_Project_ID(self,
                         ID_condidates):
        system_setting_prompt = '''
You are tasked with analyzing a list of extracted text snippets from a research paper. Each snippet includes a database ID and some surrounding context. Your job is to determine which of these IDs are associated with original data of metagenome analysis or 16S rRNA gene amplicon (meta16S) analysis, which is generated by the study described in the paper. It's important to note that research papers often mention multiple types of IDs, including those for original data generated by the authors and those referencing previously published data used for comparative analysis or information synthesis.

When analyzing each snippet, consider the following criteria to determine if an ID is associated with the paper's original data:
+ The context explicitly mentions the collection, generation, or production of data by the authors.
+ The snippet includes language suggesting the data is novel or previously unpublished.
+ References to the process of submitting data to a database, implying the authors are the data's originators.
+ Absence of phrases that typically indicate the data is from external sources, such as 'previously published', 'obtained from', or 'sourced from'.

If one or more IDs are identified as meeting the criteria, return them as a JSON array:
```json
{
  "result": ["PRJDB4038", "PRJDB4039"]
}
```

If none of the IDs meet the criteria, return empty list:
```json
{
  "result": []
}
```
'''
        user_input = json.dumps(ID_condidates)
        return self.openai_wrapper(system_setting_prompt=system_setting_prompt,
                                   user_input=user_input)

    def generate_description_of_newly_added_keys(self,
                                                 newly_added_keys=[],
                                                 paper_content=''):
        system_setting_prompt = '''
Given the following two sets of data:
1. A list of newly extracted keys from human gut microbiome research papers that are not present in the existing database:
2. The abstract and methods sections of the research paper from which the keys were extracted:        

Your task is to generate a concise and informative description for each of the newly extracted keys that represent metadata about the study subjects or samples, considering the context provided by the abstract and methods sections of the research paper. 

Please exclude keys that represent the results of the study, such as:
- Sequencing statistics (e.g., sequencing depth, human genome mapping rate)
- Assembly statistics (e.g., number of MAGs, number of contigs, N50)
- Metagenomic taxonomic composition (e.g., abundance of bacterial taxa)
- Metagenomic functional composition (e.g., abundance of genes or pathways)
- Diversity indices (e.g., alpha diversity, beta diversity)
- Relative abundance of specific bacterial groups

Instead, focus on keys that provide information about the characteristics of the subjects or samples, such as:
- Demographic information (e.g., age, gender, ethnicity)
- Clinical information (e.g., health status, diagnosed diseases)
- Lifestyle factors (e.g., diet, smoking status, exercise habits)
- Sample type (e.g., stool, biopsy, mucosa)
- Sample collection details (e.g., collection method, storage conditions)

For each relevant key, provide a clear and concise description that accurately represents the meaning of the key in the context of human gut microbiome research. If a key is an abbreviation or a domain-specific term, use the information from the abstract and methods sections to infer its meaning and provide a description that is easily understandable to a general audience.

Keep the descriptions brief, typically one sentence or a short phrase, while still conveying the essential information about the key. If there is limited information in the abstract and methods sections about a particular key, make your best effort to infer its meaning based on the available context and general knowledge related to human gut microbiome research.

Output the descriptions as a JSON object with the following format:
{
  "key1": "description1",
  "key2": "description2",
  ...
}

Please provide your descriptions in the specified JSON format.
If none of the input keys represent metadata about the subjects or samples, return an empty JSON object.
'''

        user_input = f'''
Newly added keys:
{newly_added_keys}

Paper Abstract and Method text:
{paper_content}
'''

        if self.compute_num_token(user_input) > Config.MAX_TOKENS:
            # truncate input text
            user_input = self.truncate(user_input, Config.MAX_TOKENS)

        return self.openai_wrapper(system_setting_prompt=system_setting_prompt,
                                   user_input=user_input)