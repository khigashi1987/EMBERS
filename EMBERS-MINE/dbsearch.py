import logging
import re
import json
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from Bio import Entrez
import xml.etree.ElementTree as ET
import pandas as pd
from config import Config

def requests_retry_session(retries=5, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

class DBSearch():
    ###
    # データベース検索を行うクラス
    ###
    def __init__(self, llm):
        self.llm = llm
        self.context_length = 200
        self.project_id_pattern_1 = re.compile(r'(?:PRJD|PRJE|PRJN)[A-Z]\d+')
        self.project_id_pattern_2 = re.compile(r'(?:DRP|ERP|SRP)\d+')
        self.project_id_pattern_3 = re.compile(r'(?:DRA|ERA|SRA)\d+')
        Entrez.email = Config.USER_EMAIL

    def extract_project_id(self, text):
        logging.info('\t\tExtracting Project ID...')
        # プロジェクトIDを抽出する処理
        match1 = re.findall(self.project_id_pattern_1, text)
        match2 = re.findall(self.project_id_pattern_2, text)
        match3 = re.findall(self.project_id_pattern_3, text)

        all_matches = match1 + match2 + match3
        if len(set(all_matches)) == 1:
            project_ids = [all_matches[0]]
        elif len(set(all_matches)) > 1:
            # multiple candidates for project_id
            matches = []
            if match1:
                for match in re.finditer(self.project_id_pattern_1, text):
                    start, end = match.start(), match.end()
                    context = text[max(0, start - self.context_length):min(len(text), end + self.context_length)]
                    matches.append({'ID':text[start:end], 'context':context})
            if match2:
                for match in re.finditer(self.project_id_pattern_2, text):
                    start, end = match.start(), match.end()
                    context = text[max(0, start - self.context_length):min(len(text), end + self.context_length)]
                    matches.append({'ID':text[start:end], 'context':context})
            if match3:
                for match in re.finditer(self.project_id_pattern_3, text):
                    start, end = match.start(), match.end()
                    context = text[max(0, start - self.context_length):min(len(text), end + self.context_length)]
                    matches.append({'ID':text[start:end], 'context':context})
            llm_judge_result = json.loads(self.llm.judge_Project_ID(matches))
            project_ids = llm_judge_result['result']
        else:
            # no candidate for project_id
            project_ids = []
        logging.info(f'\t\t\tProject ID: {project_ids}')
        logging.info('\t\tExtracting Project ID...Done')
        return project_ids
    
    def samples_from_projectid_from_ENA(self, project_id):
        # Fetch sample accession numbers associated with the BioProject ID with retries
        file_report_url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={project_id}&result=read_run&fields=sample_accession"
        response = requests_retry_session().get(file_report_url)
        sample_accessions = response.text.split('\n')[1:]  # Skip header line
        sample_accessions = [line.split('\t')[0] for line in sample_accessions if line]  # Extract sample accession numbers

        # Initialize a list to hold dictionaries of sample attributes for DataFrame construction
        sample_details_list = []

        # Loop through each sample accession to fetch its XML and extract details
        for sample_acc in sample_accessions:
            sample_url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{sample_acc}"
            sample_response = requests_retry_session().get(sample_url)
            sample_xml_root = ET.fromstring(sample_response.content)

            # Extract sample details
            for sample in sample_xml_root.findall('./SAMPLE'):
                sample_dict = {'sample_accession': sample_acc}
                if 'alias' in sample.attrib:
                    sample_dict['alias'] = sample.attrib['alias']
                identifiers = sample.find('IDENTIFIERS')
                if identifiers is not None:
                    for identifier in identifiers:
                        key = identifier.tag.lower()
                        value = identifier.text
                        sample_dict[key] = value

                for attr in sample.findall('.//SAMPLE_ATTRIBUTE'):
                    tag = attr.find('TAG').text if attr.find('TAG') is not None else None
                    value = attr.find('VALUE').text if attr.find('VALUE') is not None else None
                    if tag and value:
                        sample_dict[tag] = value
        
                # Append the dictionary to the list
                sample_details_list.append(sample_dict)
        return sample_details_list

    def samples_from_projectid_from_SRA(self, project_id):
        # SRAからデータを取得
        handle = Entrez.esearch(db="sra", term=project_id, retmax=1000)
        record = Entrez.read(handle)
        handle.close()

        ids = record["IdList"]

        sample_details_list = []

        # 各IDに対して、詳細データを取得
        for sra_id in ids:
            handle = Entrez.efetch(db="sra", id=sra_id, retmode="xml")
            data = handle.read().decode('utf-8')
            # XMLデータを解析
            root = ET.fromstring(data)

            # SAMPLE タグを見つける
            sample_tag = root.find('.//SAMPLE')
            sample_dict = {}
            # SAMPLE タグ内のすべての要素を辞書に追加
            for element in sample_tag:
                if element.tag == 'IDENTIFIERS' or element.tag == 'SAMPLE_NAME':
                    for subelement in element:
                        sample_dict[subelement.tag] = subelement.text
                elif element.tag == 'SAMPLE_ATTRIBUTES':
                    # SAMPLE_ATTRIBUTES はネストされたTAGとVALUEを含む
                    for attribute in element:
                        tag = attribute.find('TAG').text
                        value = attribute.find('VALUE').text
                        # 既存のキーを無視する
                        if tag not in sample_dict:
                            sample_dict[tag] = value
                else:
                    sample_dict[element.tag] = element.text

            # Append the dictionary to the list
            sample_details_list.append(sample_dict)

        return sample_details_list

    def samples_from_projectid_list(self, Project_IDs):
        sample_dataframes = []
        for project_id in Project_IDs:
            try:
                logging.info(f'\tFetching sample details for project ID {project_id} from SRA...')
                sample_details_list = self.samples_from_projectid_from_SRA(project_id)
                # Convert the list of dictionaries to a DataFrame
                df_samples = pd.DataFrame(sample_details_list)
                sample_dataframes.append(df_samples)
                logging.info(f'\tFetching sample details for project ID {project_id} from SRA...Done')
            except Exception as e:
                try:
                    logging.info(f'\tFetching sample details for project ID {project_id} from ENA...')
                    sample_details_list = self.samples_from_projectid_from_ENA(project_id)
                    # Convert the list of dictionaries to a DataFrame
                    df_samples = pd.DataFrame(sample_details_list)
                    sample_dataframes.append(df_samples)
                    logging.info(f'\tFetching sample details for project ID {project_id} from ENA...Done')
                except Exception as e:
                    logging.error(f'Failed to fetch sample details for project ID {project_id} from ENA and SRA.')
                    print(e)
        if len(sample_dataframes) == 0:
            return []
        elif len(sample_dataframes) == 1:
            return sample_dataframes[0]
        else:
            return pd.concat(sample_dataframes, ignore_index=True, sort=False)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python dbsearch.py project_id')
        sys.exit(1)
    
    project_id = sys.argv[1]

    llm = None
    dbsearch = DBSearch(llm)
    print(project_id)
    if ',' in project_id:
        project_id = project_id.split(',')
    else:
        project_id = [project_id]
    samples = dbsearch.samples_from_projectid_list(project_id)
    for c in samples.columns:
        print(c)
        print('\t\t', samples[c].values[:5])
    print(samples)




