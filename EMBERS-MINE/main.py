import os
import glob
import pickle
import logging
import pandas as pd
from io import StringIO
from config import Config
from llm import LLM
from xmlloader import XMLLoader
from dbsearch import DBSearch
from excelloader import EXCELLoader
from suppmatloader import SUPPMATLoader
import utils
import json
import datetime

class Analyzer():
    ###
    # PMCのディレクトリにあるxmlファイル、excelファイル、およびサプリPDFファイルを読み込み、
    # プロジェクト、メソッド、サンプルのメタデータを抽出する
    ###
    def __init__(self, llm, xmlloader, dbsearch, excelloader, suppmatloader):
        self.llm = llm
        self.xmlloader = xmlloader
        self.dbsearch = dbsearch
        self.excelloader = excelloader
        self.suppmatloader = suppmatloader

        self.main_content = ''
        self.main_content_truncated = ''
        self.abstract_content = ''
        self.method_content = ''

        self.excel_contents = []
        self.suppmat_contents = []
    
    def load_xml(self, pmc_dir):
        # xmlファイルのテキスト抽出
        logging.info('\tLoading xml file...')

        abstract_content_path = os.path.join(pmc_dir, 'abstract_content.pkl')
        method_content_path = os.path.join(pmc_dir, 'method_content.pkl')
        main_content_path = os.path.join(pmc_dir, 'main_content.pkl')
        if os.path.exists(abstract_content_path) and\
            os.path.exists(method_content_path) and\
                os.path.exists(main_content_path):
            with open(abstract_content_path, 'rb') as f:
                self.abstract_content = pickle.load(f)
            with open(method_content_path, 'rb') as f:
                self.method_content = pickle.load(f)
            with open(main_content_path, 'rb') as f:
                self.main_content = pickle.load(f)
        else:
            self.abstract_content = self.xmlloader.analyze_abstract(pmc_dir)
            self.method_content = self.xmlloader.analyze_materials_and_methods(pmc_dir)
            self.main_content = self.xmlloader.analyze_main(pmc_dir)
            # write to pickle
            with open(abstract_content_path, 'wb') as f:
                pickle.dump(self.abstract_content, f)
            with open(method_content_path, 'wb') as f:
                pickle.dump(self.method_content, f)
            with open(main_content_path, 'wb') as f:
                pickle.dump(self.main_content, f)

        # トークン数がMAX_TOKENSを超える場合は、truncateする
        n_token_paper = self.llm.compute_num_token(text=self.main_content)
        logging.info(f'\t\tNumber of tokens in paper: {n_token_paper}')

        if n_token_paper > Config.MAX_TOKENS:
            self.main_content_truncated = self.llm.truncate(input_text=self.main_content, max_tokens=Config.MAX_TOKENS)
        else:
            self.main_content_truncated = self.main_content

        logging.info('\tLoading xml file...Done.')
    
    def load_excel(self, pmc_dir):
        # excelファイルのテーブル抽出
        logging.info('\tLoading excel file...')

        excel_contents_path = os.path.join(pmc_dir, 'excel_contents.pkl')
        if os.path.exists(excel_contents_path):
            with open(excel_contents_path, 'rb') as f:
                self.excel_contents = pickle.load(f)
        else:
            self.excel_contents = self.excelloader.analyze_excel(pmc_dir)
            with open(excel_contents_path, 'wb') as f:
                pickle.dump(self.excel_contents, f)

        logging.info('\tLoading excel file...Done.')
    
    def load_suppmat(self, pmc_dir):
        # Suppメソッドの抽出、および
        # pdfファイルのテーブル抽出（サプリメンタリーテーブルの抽出）
        logging.info('\tLoading supplementary materials file...')

        method_contents_path = os.path.join(pmc_dir, 'method_content.pkl')
        suppmat_contents_path = os.path.join(pmc_dir, 'suppmat_contents.pkl')
        if os.path.exists(suppmat_contents_path):
            with open(suppmat_contents_path, 'rb') as f:
                self.suppmat_contents = pickle.load(f)
        else:
            supp_methods, supp_tables = self.suppmatloader.analyze_suppmat(pmc_dir)
            self.method_content += '\nSupplementary Methods:\n' + supp_methods
            self.suppmat_contents = supp_tables
            with open(method_contents_path, 'wb') as f:
                pickle.dump(self.method_content, f)
            with open(suppmat_contents_path, 'wb') as f:
                pickle.dump(self.suppmat_contents, f)

        logging.info('\tLoading pdf file...Done.')

    def analyze_pmc(self, pmc_dir, log_prefix, out_prefix):
        working_out_file = f'{log_prefix}_working.txt'
        ofp = open(working_out_file, 'w')

        try:
            self.load_xml(pmc_dir)
        except Exception as e:
            print(f'XML Loading Error: {e}')
            ofp.write(f'XML Loading Error: {e}\n')
            return
        # Table loading is a heavy process and should be postponed as long as possible.

        run_processes = {
            "Step1": True,
            "Step2": True,
            "Step3": True,
            "Step4": True,
            "Step5": True,
            "Step6": True
        }

        # 1. Determine the type of the study from the abstract
        if run_processes['Step1']:
            logging.info('\tDetermine the type of the study from the abstract...')
            if len(self.abstract_content) < 10:
                logging.info('\t\tAbstract not found. Skip the process.')
                ofp.close()
                return
            result = self.llm.determine_target_study_or_not(abstract_text=self.abstract_content,
                                                            method_text=self.method_content)
            result = json.loads(result)
            ofp.write(f'Determined the type of the study from the abstract:\n{result}\n\n')
            if result['decision'] == 'no':
                # skip the rest of the process
                logging.info('\t\tThe study is not a target study. Skip the process.')
                ofp.close()
                return
            logging.info('\tDetermine the type of the study from the abstract...Done.')

        # Loading Supplementary materials (docx, pdf)
        self.load_suppmat(pmc_dir)

        # 2. Extract project information from the paper
        if run_processes['Step2'] and\
            not os.path.exists(f'{out_prefix}_project.json'):
            logging.info('\tExtract project information from the paper...')
            schema = json.load(open(Config.SCHEMA_PROJECT_JSON))
            result = self.llm.analyze_project_info(schema=schema, 
                                                   abstract_text=self.abstract_content, 
                                                   method_text=self.method_content)
            result = json.loads(result)
            updated_schema = utils.update_project_schema(result)
            with open(Config.SCHEMA_PROJECT_JSON, 'w') as f:
                json.dump(updated_schema, f, indent=4)
            ofp.write(f'\nExtract project information from the paper:\n{result}\n\n')
            utils.make_info_project(out_prefix, result)
            logging.info('\tExtract project information from the paper...Done.')
        
        # 3. Extract experimental protocols from the paper
        if run_processes['Step3'] and\
            not os.path.exists(f'{out_prefix}_methods.json'):
            logging.info('\tExtract experimental protocols from the paper...')
            result = self.llm.analyze_methods(method_text=self.method_content)
            result = json.loads(result)
            ofp.write(f'\nExtract experimental protocols from the paper:\n{result}\n\n')
            utils.make_info_methods(out_prefix, result)
            logging.info('\tExtract experimental protocols from the paper...Done.')

        # 4. Extract information of Public database registration
        if run_processes['Step4'] and\
            not os.path.exists(f'{out_prefix}_samples.json'):
            logging.info('\tExtract information of Public database registration...')
            project_ids = self.dbsearch.extract_project_id(self.abstract_content+'\n'+self.main_content+'\n'+self.method_content)
            if len(project_ids) > 0:
                utils.add_info_project(out_prefix, 'Project ID', project_ids)
            samples_df = self.dbsearch.samples_from_projectid_list(project_ids)
            if len(samples_df) == 0:
                # skip ther rest of the process
                logging.info('\t\tNo sample information found. Skip the process.')
                ofp.close()
                return
            utils.make_sample_list(out_prefix, samples_df)
            logging.info('\tExtract information of Public database registration...Done.')

        # Loading Excel files
        self.load_excel(pmc_dir)
        
        # 5. エクセルの各テーブル、サプリPDFを巡回してサンプル情報を更新
        if run_processes['Step5']:
            logging.info('\tExtract sample information from the tables...')
            contents = self.excel_contents +\
                        self.suppmat_contents
            for content in contents:
                content, id_column, id_key = utils.check_table_both_direction(out_prefix, content)
                ofp.write(f'\n\tTable: {content.head(5)}\n')
                ofp.write(f'\t\tID column: {id_column}\n')
                ofp.write(f'\t\tID key: {id_key}\n')
                if id_column is not None and id_key is not None:
                    utils.update_sample_list(out_prefix, content, id_column, id_key)
            logging.info('\tExtract sample information from the tables...Done.')
        
        
        # 6. Generate description of newly added sample keys
        if run_processes['Step6']:
            numerical_items, categorical_items = utils.cleanse_sample_list(out_prefix)
            
            logging.info('\tGenerate description of newly added sample keys...')
            current_keys = utils.current_keys(out_prefix)
            if len(current_keys) > 0:
                result = self.llm.generate_description_of_newly_added_keys(current_keys,
                                                                           self.abstract_content+'\n'+self.method_content)
                result = json.loads(result)
            else:
                result = {}

            ofp.write(f'\nDescription of newly added sample keys:\n{result}\n\n')
            with open(f'{out_prefix}_new_keys_descriptions.json', 'w') as f:
                json.dump(result, f, indent=4)

            logging.info('\tGenerate description of newly added sample keys...Done.')
        
        ofp.close()
        return


if __name__ == '__main__':
    ### setup_logging()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(os.path.join(Config.LOG_DIR, f'log_{current_time}.txt'))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    ###

    logging.info('Start analyzing process...')

    # Validation dataset
    TARGET_PMCs = [os.path.basename(pmcdir) for pmcdir in glob.glob(os.path.join(Config.PMC_DIR, 'PMC*'))]

    for i, TARGET_PMC in enumerate(TARGET_PMCs):

        logging.info(f'{i} Analyzing PMC: {TARGET_PMC}')

        # Directories
        pmc_dir = os.path.join(Config.PMC_DIR, f'{TARGET_PMC}')
        result_dir = os.path.join(Config.RESULT_BASE_DIR, f'{TARGET_PMC}')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if os.path.exists(os.path.join(result_dir, 'finished_analysis')):
            # already analyzed
            logging.info(f'\tAlready analyzed. Skip the process.')
            continue

        out_prefix = os.path.join(result_dir, f'{TARGET_PMC}')
        log_prefix = os.path.join(Config.LOG_DIR, f'{TARGET_PMC}')

        # Salvage data which was skipped in previous attempts
        if os.path.exists(f'{out_prefix}_project.json'):
            # not skipped data (analyzed in previous attempts)
            logging.info(f'\tAlready analyzed. Skip the process.')
            continue

        # Initialize
        llm = LLM(api_key=Config.OPENAI_API_KEY, model_name=Config.MODEL_NAME)
        xmlloader = XMLLoader()
        excelloader = EXCELLoader()
        suppmatloader = SUPPMATLoader()
        dbsearch = DBSearch(llm=llm)
        analyzer = Analyzer(llm=llm,
                            xmlloader=xmlloader,
                            dbsearch=dbsearch,
                            excelloader=excelloader,
                            suppmatloader=suppmatloader)

        # Analyze
        analyzer.analyze_pmc(pmc_dir, log_prefix, out_prefix)

        logging.info(f'End Analyzing PMC: {TARGET_PMC}\n\n')
        with open(os.path.join(result_dir, 'finished_analysis'), 'w') as f:
            f.write('')

    logging.info('End analyzing process.')