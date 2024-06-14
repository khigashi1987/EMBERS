import openai
import tiktoken

class LLM():
    ###
    # LLM関連の処理を実行するクラス
    ###
    def __init__(self, 
                 api_key='', 
                 model_name='gpt-4-turbo',
                 max_tokens=2048):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
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
    
    def generate_transformation_code(self,
                                     input_conditions=[]):
        system_setting_prompt = '''
Given the following inputs:
1. Reference key list and target key
2. Descriptions of the reference keys
3. Description of the target key
4. List of sample values

Your task is to generate Python code that takes a single sample dict record as input (variable name: input) and transforms it to the target key value.

You should output a JSON object with the following format:

{"Conversion_possible": "yes" or "no",
"Python_code": "A Python script that can be directly executed using exec(). The script should define a function named 'transform_data' that takes the 'input' variable as its argument, performs the data transformation, and returns only the transformed value (not a dictionary or any other data structure). The function should handle errors and edge cases gracefully.",
"Reason": "Explanation of the approach used in the Python code"}

The Python code should attempt to convert as many sample records as possible based on the data types, patterns, and value ranges observed in the provided samples. The code should handle various data types (integers, floats, strings) and normalize the values to the desired format specified in the target key description.
Include error handling and edge case considerations in the generated code. Provide clear comments explaining the logic behind the conversion process.

Please ensure that the generated Python code only defines the 'transform_data' function and does not include any example usage or test cases. The 'transform_data' function should be directly callable with the 'input' variable after executing the script with exec(), and it should return only the transformed value.
'''

        user_input = f'''
Reference key list:
{input_conditions['reference_keys']}
Target key name:
{input_conditions['target_key']}

Descriptions of the reference keys:
{input_conditions['reference_key_descriptions']}

Description of the target key:
{input_conditions['target_key_description']}

List of sample values:
{input_conditions['sample_values']}
'''
        return self.openai_wrapper(system_setting_prompt=system_setting_prompt,
                                   user_input=user_input)
