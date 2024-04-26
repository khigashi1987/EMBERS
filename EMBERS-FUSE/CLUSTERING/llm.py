import openai
import tiktoken

class LLM():
    ###
    # LLM関連の処理を実行するクラス
    ###
    def __init__(self, 
                 api_key='', 
                 model_name='text-embedding-3-large',
                 max_tokens=2048):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
    
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