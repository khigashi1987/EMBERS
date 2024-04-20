import os
import glob
import logging
import itertools
from lxml import etree

class XMLLoader():
    ###
    # PMC論文のxmlファイルを読み込み、テキストを抽出するクラス
    ###
    def __init__(self):
        materials_patterns = ['material', 'materials']
        and_patterns = ['and', '&', '&#x00026;']
        methods_patters = ['method', 'methods']
        other_patterns = ['methods summary',
                          'experimental procedures',
                          'patients and methods',
                          'experimental design, materials and methods',
                          'methods and materials']
        self.methods_title_examples = other_patterns + \
                                      materials_patterns + \
                                      methods_patters + \
                                      [' '.join([m1, m2, m3]) for m1, m2, m3 in itertools.product(materials_patterns, and_patterns, methods_patters)]

    def extract_element_text(self, element):
        '''
        if element.text:
            text = element.text
        else:
            text = " "
        for child in element:
            text += " " + self.extract_element_text(child)
        return text
        '''
        return ' '.join(element.itertext())
    
    def check_title_means_methods(self, title):
        if title is None or title.text is None:
            return False
        title = title.text.lower()

        if '. ' in title:
            # e.g. "2. Materials and Methods"
            title = title.split('. ')[1].strip()

        if title in self.methods_title_examples:
            return True
        return False
    
    def get_abstract(self, root):
        abstract = root.find(".//abstract")
        if abstract is not None:
            return self.extract_element_text(abstract)

        abstract = root.find(".//sec[title='Abstract']")
        if abstract is not None:
            return self.extract_element_text(abstract)

        return "Abstract"  # not found

    def get_materials_and_methods(self, root):
        method_content_str = ""

        # Data Availability Statement for PLoS journals
        data_availability = root.xpath(".//custom-meta[@id='data-availability']")
        if data_availability:
            method_content_str += f"Data Availability Statement\n"
            method_content_str += self.extract_element_text(data_availability[0])
            method_content_str += "\n"
        
        # Data Availability Statement for Scientific Reports
        data_availability = root.xpath(".//notes[@notes-type='data-availability']")
        if data_availability:
            method_content_str += f"Data Availability Statement\n"
            method_content_str += self.extract_element_text(data_availability[0])
            method_content_str += "\n"
        
        # Data Availability Statement written in the Notes section
        notes = root.findall(".//notes")
        for nt in notes:
            method_content_str += "\nSection Title: Notes\n"
            method_content_str += self.extract_element_text(nt)
            method_content_str += "\n"

        # Data Availability Statement written in the footnote
        footnotes = root.findall(".//fn")
        for fn in footnotes:
            method_content_str += "\nSection Title: Footnote\n"
            method_content_str += self.extract_element_text(fn)
            method_content_str += "\n"
        
        # Data Availability Statement written in the acknowledgements
        acknowledgements = root.findall(".//ack")
        for ack in acknowledgements:
            method_content_str += "\nSection Title: Acknowledgements\n"
            method_content_str += self.extract_element_text(ack)
            method_content_str += "\n"

        sections = root.findall(".//sec")
        for sec in sections:
            title = sec.find("title")
            parent_title = sec.getparent().find("title")

            # Extract only Materials and Methods section
            if self.check_title_means_methods(title):
                method_content_str += f"\nSection Title: {title.text}\n"
                method_content_str += self.extract_element_text(sec)
                method_content_str += "\n"
            elif self.check_title_means_methods(parent_title):
                # Skip Children blocks of Materials and Methods section
                # (because they are already included in the parent section)
                continue

        for sec in root.findall(".//sec[@sec-type='materials|methods']"):
            method_content_str += self.extract_element_text(sec)

        for sec in root.findall(".//sec[@sec-type='methods']"):
            method_content_str += self.extract_element_text(sec)

        return method_content_str

    def get_main_content(self, root):
        main_content_str = ""

        # body直下のpタグを取得（for NIHMS論文）
        paragraphs = root.findall(".//body/p")
        for p in paragraphs:
            main_content_str += self.extract_element_text(p)
            main_content_str += "\n"

        # ほかの論文はたいてsecタグ以下にメインテキストが配置されてる
        sections = root.findall(".//sec")
        for sec in sections:
            title = sec.find("title")
            parent_title = sec.getparent().find("title")

            if self.check_title_means_methods(title):
                # Skip Materials and Methods section
                continue
            elif self.check_title_means_methods(parent_title):
                # Skip Children blocks of Materials and Methods section
                continue
            elif title is not None:
                main_content_str += f"\nSection Title: {title.text}\n"

            main_content_str += self.extract_element_text(sec)
            main_content_str += "\n"

        return main_content_str
    
    def analyze_abstract(self, pmc_dir):
        article_xml_list = glob.glob(os.path.join(pmc_dir, '*.nxml'))
        extracted_text = ''
        for xml_file in article_xml_list:
            root = etree.parse(xml_file).getroot()
            extracted_text += self.get_abstract(root)
        return extracted_text

    def analyze_materials_and_methods(self, pmc_dir):
        article_xml_list = glob.glob(os.path.join(pmc_dir, '*.nxml'))
        extracted_text = ''
        for xml_file in article_xml_list:
            root = etree.parse(xml_file).getroot()
            extracted_text += self.get_materials_and_methods(root)
        return extracted_text

    def analyze_main(self, pmc_dir):
        article_xml_list = glob.glob(os.path.join(pmc_dir, '*.nxml'))
        extracted_text = ''
        for xml_file in article_xml_list:
            ###
            # TODO: 複数xmlファイルがある場合の対処
            ###
            logging.info(f'\t\tanalyzing {xml_file}')

            root = etree.parse(xml_file).getroot()
            extracted_text += self.get_main_content(root)

            logging.info(f'\t\tanalyze done. {xml_file}')
        return extracted_text
    

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <PMC directory>')
        sys.exit(1)
    
    pmc_dir = sys.argv[1]
    xml_loader = XMLLoader()

    print('*****************************')
    print('\nAbstract:\n')
    abstract_text = xml_loader.analyze_abstract(pmc_dir)
    print(abstract_text, '\n\n')

    print('*****************************')
    print('\nMaterials and Methods:\n')
    methods_text = xml_loader.analyze_materials_and_methods(pmc_dir)
    print(methods_text, '\n\n')

    print('*****************************')
    print('\nMain Content:\n')
    main_text = xml_loader.analyze_main(pmc_dir)
    print(main_text, '\n\n')