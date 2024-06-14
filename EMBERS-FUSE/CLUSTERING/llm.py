import openai
import tiktoken

class LLM():
    ###
    # LLM関連の処理を実行するクラス
    ###
    def __init__(self, 
                 api_key='', 
                 completion_model_name='gpt-4-turbo',
                 max_tokens=2048):
        self.client = openai.OpenAI(api_key=api_key)
        self.completion_model_name = completion_model_name
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.encoding_for_model(self.completion_model_name)

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
                model=self.completion_model_name,
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

    def summarize_project_findings(self,
                                   project_findings=[]):
        system_setting_prompt = '''
Please analyze the following key findings from research papers focusing on the human gut microbiome. Please identify the common theme or focus shared by these studies and express it in the shortest possible English word or phrase. Avoid vague classifications that apply to most papers, such as "human gut microbiome research." Instead, focus on the specific elements that are unique to this cluster of papers.

Please provide your output in the following JSON format:
Output format:
{
  "Topic": "Estimated topic representing the cluster of papers",
  "Reason": "Explanation for the chosen topic"
}

Example:
['The gut microbiota composition and function in diabetic retinopathy patients differed significantly from healthy controls, with specific bacterial genera identified as potential non-invasive biomarkers.',
'The study identified distinct gut microbiota compositions and fecal metabolic phenotypes in diabetic retinopathy patients compared to diabetic controls, with specific microbial pathways providing potential etiological and therapeutic targets.',
'Gut microbiota alterations in DCI patients compared to T2DM patients are accompanied by changes in SCFAs and inflammatory cytokines, potentially influencing DCI development.']

Example output:
{
  "Topic": "Gut microbiome alterations linked to diabetic complications",
  "Reason": "The key findings consistently highlight the association between gut microbiome changes and various diabetic complications, such as retinopathy and cognitive impairment (DCI)."
}
'''

        project_findings_text = '\n'.join(project_findings)
        user_input = f'''
Key Findings:
{project_findings_text}
'''
        return self.openai_wrapper(system_setting_prompt=system_setting_prompt,
                                   user_input=user_input)

    def summarize_methods(self,
                          methods_texts=[]):
        system_setting_prompt = '''
Please analyze the following set of text snippets that describe nearly identical experimental procedure steps in human gut microbiome research. These snippets have been clustered together based on their similarity.
Your task is to create a concise label in English (preferably a short phrase or a single word) that best represents the common experimental procedure described by this cluster. The label should be more specific and have a higher resolution than generic terms like "Fecal Sample Collection" or "DNA Extraction," as all the input texts will fall under one of these broad categories. Aim to capture the unique aspects of the procedure described in the cluster.
Additionally, provide a brief explanation of why you think this label effectively represents the cluster.

Please provide your output in the following JSON format:
{
  "Label": "Short English phrase or word representing the cluster",
  "Reason": "Brief explanation of why this label represents the cluster"
}

Example input (a cluster of text snippets):
[
  "DNA was extracted from fecal samples using the QIAamp DNA Stool Mini Kit (Qiagen) according to the manufacturer's instructions.",
  "Fecal DNA was isolated using the QIAamp DNA Stool Mini Kit (Qiagen, Hilden, Germany) following the manufacturer's protocol.",
  "Bacterial genomic DNA was extracted from stool samples using the QIAamp DNA Stool Mini Kit (Qiagen, Germany) as per the manufacturer's instructions."
]

Example output:
{
  "Label": "DNA extraction using QIAamp DNA Stool Mini Kit",
  "Reason": "All the snippets in this cluster describe the process of extracting DNA from fecal samples using the same commercially available kit (QIAamp DNA Stool Mini Kit) and following the manufacturer's instructions. This label captures the specific kit used, which differentiates this cluster from other DNA extraction methods."
}
'''

        methods_text = "[" + '\n'.join(['"'+t+'"' for t in methods_texts]) + "]"
        user_input = f'''
Cluster of snippets of experimental procedure steps in human gut microbiome research:
{methods_text}
'''
        return self.openai_wrapper(system_setting_prompt=system_setting_prompt,
                                   user_input=user_input)

    def calculate_purity(self,
                         query_texts=[]):
        system_setting_prompt = '''
You will be provided with a list of metadata fields extracted from research papers on the human gut microbiome. Each metadata field will be represented by a text string containing the following information:

1. The original key name of the metadata field (as used in the source paper's data table)
2. A brief description of the metadata field
3. A few unique example values randomly selected from the original data table

Your task is to analyze the semantic content of these metadata fields and evaluate the "semantic purity" of the set as a whole. "Semantic purity" refers to the degree to which the fields all pertain to the same specific metadata concept and share the same data type (e.g., string, numeric, date), as opposed to a mixture of different metadata concepts or data types.

Please provide your evaluation of the semantic purity of the provided set of metadata fields as a score between 0.0 and 1.0, where 1.0 indicates the fields are highly semantically homogeneous (i.e., they all describe the same metadata concept and have the same data type) and 0.0 indicates the fields are highly semantically heterogeneous (i.e., they describe a variety of different metadata concepts or have different data types).

For example, if provided with the following set of metadata fields:

"Key: Age Description: age of subject Examples: [16, 21, 65, 24, 54]"
"Key: host_age Description: subject age Examples: [52, 11, 83, 43, 65]" 
"Key: age(years) Description: age in years Examples: [92, 91, 86, 93, 85]"

The semantic purity score should be close to 1.0, as the key names, descriptions, and example values all clearly refer to the same concept of subject age and have the same data type (numeric).

Also, if provided with a set of metadata fields like:

"Key: Gender Description: gender of subject Examples: ['F', 'M']"
"Key: host_sex Description: biological sex of study subjects Examples:['male', 'female']"

The semantic purity score should be close to 1.0, as the key names, descriptions, and example values all clearly refer to the same concept of subject sex and have the same data type (string).

By contrast, if provided with a set of metadata fields like:

"Key: Age Description: age of subject Examples: [16, 21, 65, 24, 54]"
"Key: Gender Description: sex of subject Examples: [Male, Female, Female, Male, Female]"
"Key: BMI Description: body mass index Examples: [24.3, 18.9, 27.1, 23.5, 26.4]"
"Key: Collection_Date Description: sample collection date Examples: [2022-03-15, 2021-11-02, 2023-01-28, 2022-09-10, 2022-05-23]"

The semantic purity score should be closer to 0.0, as these fields refer to a semantically heterogeneous set of metadata concepts and have different data types (numeric for age and BMI, string for gender, and date for collection date).

Similarly, if provided with metadata fields like:

"Key: Country Description: geographical location name (country or sea) Examples: [USA, Japan, France, China, Australia]"
"Key: Latitude_Longitude Description: latitude and longitude Examples: [(40.7128, -74.0060), (35.6895, 139.6917), (48.8566, 2.3522), (39.9042, 116.4074), (-33.8688, 151.2093)]"

The semantic purity score should be lower, as these fields, while related to geographic location, likely have different data types (string for country names and numeric for latitude and longitude).

On the other hand, if provided with the following set of metadata fields:

"Key: Age Description: age of subject Examples: [16, 21, 65, 24, 54]"
"Key: host_age Description: subject age Examples: [52, 11, 83, 43, 65]"
"Key: age(years) Description: age in years Examples: [92.0, 91.0, 86.0, 93.0, 85.0]"
"Key: age_months Description: subject age in months Examples: ['3.0 months', '6.0 months', '18.0 months', '9.0 months', '12.0 months']"

The semantic purity score should be higher, as all fields clearly refer to the same concept of subject age, despite differences in data type (integer, float, string) and unit of measurement (years, months). It's because that Years-Months transformation can be easily done and the string values in "age_months" can be easily parsed to obtain numeric values.

Please provide the semantic purity score and a brief explanation of your reasoning in the specified JSON format:

{"Purity": [SEMANTIC PURITY SCORE],
"Reasoning": "[BRIEF EXPLANATION OF THE REASONING BEHIND THE PURITY SCORE]"}
'''

        query_texts_string = ' ,\n'.join(query_texts)
        user_input = f'''
List of descriptions:
[
{query_texts_string}
]
'''

        # if user_input is too long, truncate it
        if self.compute_num_token(user_input) > self.max_tokens:
                user_input = self.truncate(user_input, self.max_tokens)

        return self.openai_wrapper(system_setting_prompt=system_setting_prompt,
                                   user_input=user_input)

    def summarize_keys(self,
                descriptions=[]):
        system_setting_prompt = '''
You will be provided with a list of text descriptions that have been identified as a highly semantically pure set, meaning they all pertain to the same specific metadata concept, share the same data type, and have the same unit of measurement (if applicable). These descriptions are associated with metadata fields extracted from research papers on the human gut microbiome, describing characteristics of the study subjects or biological samples.
Your task is to generate a concise and informative label for this metadata item, as well as a brief description of what the metadata item represents in the context of human gut microbiome research. The label should be a clear and standardized name for the metadata item, while the description should provide additional context about the meaning and use of the metadata item.

Please provide the label and description in the following JSON format:

{"Label": "[METADATA ITEM LABEL]",
 "Description": "[METADATA ITEM DESCRIPTION]"}

 For example, if provided with the following set of descriptions:

"age of subject (years)"
"subject age (years)"
"age in years"

The output might be:

{"Label": "Age (years)",
 "Description": "Metadata item describing the age of the study subjects in years"}
'''

        descriptions_string = ' ,\n'.join(descriptions)
        user_input = f'''
List of descriptions:
[
{descriptions_string}
]
'''
        # if user_input is too long, truncate it
        if self.compute_num_token(user_input) > self.max_tokens:
            user_input = self.truncate(user_input, self.max_tokens)

        return self.openai_wrapper(system_setting_prompt=system_setting_prompt,
                                   user_input=user_input)