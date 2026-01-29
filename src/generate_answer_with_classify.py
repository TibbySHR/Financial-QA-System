import os
import json
import copy
import parse
import re
from itertools import chain
from loguru import logger
from datetime import datetime

import re_util
from config import cfg
from file import load_total_tables
from file import load_tables_of_years
from file import add_growth_rate_in_table
from file import table_to_text, add_text_compare_in_table
from file import load_pdf_info, load_test_questions
from company_table import get_sql_search_cursor, load_company_table
from recall_report_text import recall_annual_report_texts
from recall_report_names import recall_pdf_tables
from chatglm_ptuning import ChatGLM_Ptuning
import type2, type1
import prompt_util
import question_util
import sql_correct_util

'''
对测试集问题进行分类，并将结果保存到csv文件中。
'''
def do_classification(model: ChatGLM_Ptuning):
    logger.info('Do classfication...')
    # 加载测试问题集
    test_questions = load_test_questions()  
    # 加载 PDF 信息
    pdf_info = load_pdf_info()
    # 构建分类结果保存路径
    classify_dir = os.path.join(cfg.DATA_PATH, 'classify')
    # 检查目录是否存在
    if not os.path.exists(classify_dir):
        os.mkdir(classify_dir)
    # 遍历每一个问题
    for question in test_questions:
        # 生成结果文件路径
        class_csv = os.path.join(classify_dir, '{}.csv'.format(question['id']))
        # 匹配问题中的公司名称
        mactched_comp_names = question_util.get_match_company_names(question['question'], pdf_info)

        # 带颜色的日志输出
        logger.opt(colors=True).info('<blue>Start process question {} {}</>'.format(question['id'], question['question']))
        # 将问题输入模型，得到模型的输出结果
        result = model.classify(question['question'])
        
        # 规则1：若问题包含特定关键词，强制设为 F 类
        if re.findall('(状况|简要介绍|简要分析|概述|具体描述|审计意见)', question['question']):
            result = 'F'
        
        # 规则2：若问题包含定义类关键词，强制设为 F 类
        if re.findall('(什么是|指什么|什么意思|定义|含义|为什么)', question['question']):
            result = 'F'

        # 规则3：若结果为 A-D 但未匹配到公司，改为 F 类
        if result in ['A', 'B', 'C', 'D'] and len(mactched_comp_names) == 0:
            logger.info('AAAA{}'.format(question['question']))
            result = 'F'
        
        # 规则4：若结果为 E 但匹配到公司，改为 G 类
        if result in ['E'] and len(mactched_comp_names) > 0:
            logger.info('BBBBB{}'.format(question['question']))
            result = 'G'

        # 清理结果中的 < 字符
        logger.info(result.replace('<', ''))

        with open(class_csv, 'w', encoding='utf-8') as f:
            save_result = copy.deepcopy(question)   # 深拷贝原始问题数据
            save_result['class'] = result    # 添加分类结果字段

            json.dump(save_result, f, ensure_ascii=False)   # 以 JSON 格式写入 CSV 文件

'''
生成测试集的关键词，并保存到csv文件中
'''
def do_gen_keywords(model: ChatGLM_Ptuning):
    # 记录关键词生成任务开始日志
    logger.info('Do gen keywords...')
    # 加载测试问题集
    test_questions = load_test_questions()

    pdf_info = load_pdf_info()
    
    # 构建关键词结果保存路径
    keywords_dir = os.path.join(cfg.DATA_PATH, 'keywords')
    if not os.path.exists(keywords_dir):
        os.mkdir(keywords_dir)
    # 遍历每个问题
    for question in test_questions:
        keywords_csv = os.path.join(keywords_dir, '{}.csv'.format(question['id']))
        logger.opt(colors=True).info('<blue>Start process question {} {}</>'.format(question['id'], question['question']))
        # 生成关键词并分割为列表
        result = model.keywords(question['question']).split(',')
        # 记录原始结果
        logger.info(result)
        # 写入文件
        with open(keywords_csv, 'w', encoding='utf-8') as f:
            save_result = copy.deepcopy(question)
            if len(result) == 0:
                logger.warning('问题{}的关键词为空'.format(question['question']))
                result = [question['question']]
            save_result['keywords'] = result

            json.dump(save_result, f, ensure_ascii=False)


'''
为测试集问题生成sql语句。
'''
def do_sql_generation(model: ChatGLM_Ptuning):
    logger.info('Do sql generation...')
    test_questions = load_test_questions()

    sql_dir = os.path.join(cfg.DATA_PATH, 'sql')
    if not os.path.exists(sql_dir):
        os.mkdir(sql_dir)
    # 遍历每一个问题
    for question in test_questions:

        sql_csv = os.path.join(sql_dir, '{}.csv'.format(question['id']))
        # 初始化sql为空
        sql = None

        # 分类结果检查
        class_csv = os.path.join(cfg.DATA_PATH, 'classify', '{}.csv'.format(question['id']))
        if os.path.exists(class_csv):
            with open(class_csv, 'r', encoding='utf-8') as f:
                class_result = json.load(f)
                question_type = class_result['class']  # 提取分类标签

            if question_type == 'E':  # 仅对分类结果为 E 的问题生成 SQL
                logger.opt(colors=True).info('<blue>Start process question {} {}</>'.format(question['id'], question['question'].replace('<', '')))
                # 模型输出问题的sql语句
                sql = model.nl2sql(question['question'])
                logger.info(sql.replace('<>', ''))
        # 将sql语句写入文件
        with open(sql_csv, 'w', encoding='utf-8') as f:
            save_result = copy.deepcopy(question)
            save_result['sql'] = sql
            json.dump(save_result, f, ensure_ascii=False)

'''

'''
def generate_answer(model):
    # 记录日志：开始加载PDF信息
    logger.info('Load pdf info...')
    # 加载PDF基本信息和表格数据
    pdf_info = load_pdf_info()
    pdf_tables = load_total_tables()

    # 加载测试问题集
    test_questions = load_test_questions()
    # 获取SQL查询游标（用于后续数据库操作）
    sql_cursor = get_sql_search_cursor()
    # 从表中获取关键词列表
    key_words = list(load_company_table().columns)
    logger.info('key_words:{}'.format(key_words))

    #  创建答案存储目录
    answer_dir = os.path.join(cfg.DATA_PATH, 'answers')
    if not os.path.exists(answer_dir):
        os.mkdir(answer_dir)

    # 遍历测试集
    for question in test_questions:
        # 读取分类结果文件
        class_csv = os.path.join(cfg.DATA_PATH, 'classify', '{}.csv'.format(question['id']))
        if os.path.exists(class_csv):
            # 读取问题分类结果
            with open(class_csv, 'r', encoding='utf-8') as f:
                class_result = json.load(f)
                question_type = class_result['class'] # 获取问题类型（A-G）
        else:
            logger.warning('分类文件不存在!')
            question_type = 'F'  # 默认分类为F

        # 读取关键词文件
        keyword_csv = os.path.join(cfg.DATA_PATH, 'keywords', '{}.csv'.format(question['id']))
        if os.path.exists(keyword_csv):
            with open(keyword_csv, 'r', encoding='utf-8') as f:
                keyword_result = json.load(f)
                question_keywords = keyword_result['keywords']
        else:
            logger.warning('关键词文件不存在!')
            question_keywords = []

        answer_csv = os.path.join(answer_dir, '{}.csv'.format(question['id']))
        # 清洗问题文本（移除括号）
        ori_question = re.sub('[\(\)（）]', '', question['question'])
        # 从问题中提取年份信息
        years = question_util.get_years_of_question(ori_question)
        # 从问题中匹配最相关的pdf名称
        mactched_pdf_names = question_util.get_match_pdf_names(ori_question, pdf_info)

        # 根据给定的PDF键名列表，从PDF信息字典中提取对应的公司名称、公司缩写和公司代码
        company_abbrs = question_util.get_company_name_and_abbr_code_of_question(mactched_pdf_names, pdf_info)

        # 设置默认回答
        answer = '经查询，无法回答{}'.format(ori_question)

        if len(company_abbrs) > 0:
            company = company_abbrs[0][0] # 公司全称
            abbr= company_abbrs[0][1]  # 公司缩写
            code = company_abbrs[0][2]  # 公司代码
            real_comp = company if company in ori_question else abbr   # 使用问题中出现的名称形式   

        # 记录处理日志
        logger.opt(colors=True).info('<blue>Start process question {} {}</>'.format(question['id'], question['question'].replace('<', '')))
        logger.opt(colors=True).info('<cyan>问题类型{}</>'.format(question_type.replace('<', '')))

        try:
            if question_type in ['A', 'B', 'C', 'G']:
                # 定义每种类型对应的表格
                table_dict = {
                    'A': ['basic_info'], # 基础信息表
                    'B': ['employee_info', 'dev_info'],   # 员工和研发信息
                    'C': ['cbs_info', 'cscf_info', 'cis_info'], # 财务相关表
                    'G': ['basic_info', 'employee_info', 'dev_info', 'cbs_info', 'cscf_info', 'cis_info'] # 所有表
                }
                # 无匹配报表pdf
                if len(company_abbrs) == 0:
                    logger.warning('匹配到了类别{}, 但是不存在报表'.format(question_type))
                else:
                    logger.info('问题关键词: {}'.format(question_keywords))

                    background = ''
                    tot_matched_rows = []
                    # 按年份处理表格数据
                    for year in years:
                        # 按年份和公司加载表格
                        pdf_table = load_tables_of_years(company, [year], pdf_tables, pdf_info)
                        background += '已知{}(简称:{},证券代码:{}){}年的资料如下:\n    '.format(company, abbr, code, year)
                        # 根据关键词召回表格行
                        matched_table_rows = []
                        # 从PDF表格数据中召回与给定关键词匹配的行数据
                        for keyword in question_keywords:
                            matched_table_rows.extend(recall_pdf_tables(keyword, [year], pdf_table, 
                                min_match_number=3, valid_tables=table_dict[question_type]))
                        # 若无匹配则取全部有效表格
                        if len(matched_table_rows) == 0:
                            for table_row in pdf_table:
                                if table_row[0] in table_dict[question_type]:
                                    matched_table_rows.append(table_row)
                        # 将表格数据转化为自然语言
                        table_text = table_to_text(real_comp, ori_question, matched_table_rows, with_year=False)
                        background += table_text  # 将表格内容添加到信息中
                        background += '\n'
                        # 保存当前年份的匹配行
                        tot_matched_rows.extend(matched_table_rows)
                    # 跨年份对比表格数据
                    tot_matched_rows = add_text_compare_in_table(tot_matched_rows)
                    # 将所有年份的匹配行转换为自然语言文本
                    tot_text = table_to_text(real_comp, ori_question, tot_matched_rows, with_year=True)

                    # 如果文本中包含“相同”或“不相同且不同”，说明表格对比结果已明确，直接将其作为答案
                    if '相同' in tot_text or '不相同且不同' in tot_text:
                        answer = tot_text
                    else:    # 否则需要调用模型生成回答。
                        # 获取提示词
                        question_for_model = type1.get_prompt(ori_question, company, abbr, years).format(background, ori_question)
                        logger.info('Prompt length {}'.format(len(question_for_model)))
                        if len(question_for_model) > 5120:
                            question_for_model = question_for_model[:5120]
                        logger.info(question_for_model.replace('<', ''))
                        # 调用模型。
                        answer = model(question_for_model)
                    logger.opt(colors=True).info('<magenta>{}</>'.format(answer.replace('<', '')))

            # 当问题类型为D时
            elif question_type == 'D':
                # 没有匹配到公司
                if len(company_abbrs) == 0:
                    logger.warning('匹配到了类别{}, 但是不存在报表'.format(question_type))
                # 匹配到了公司
                else:

                    logger.info('问题关键词: {}'.format(question_keywords))
                    # 判断是否属于要计算增长率的问题。
                    if type2.is_type2_growth_rate(ori_question):
                        # 初始化一个空列表，用于存储需要处理的年份。
                        years_of_table = []
                        # 遍历提取到的年份
                        for year in years:
                            # 将当前年份和前一年的年份添加到列表中
                            years_of_table.extend([year, str(int(year)-1)])
                        # 加载指定公司，年份的表格数据
                        pdf_table = load_tables_of_years(company, years_of_table, pdf_tables, pdf_info)
                        # 在表格信息中添加增长率行信息
                        pdf_table = add_growth_rate_in_table(pdf_table)
                    # 判断是否属于公式计算问题
                    elif type2.is_type2_formula(ori_question):
                        # 加载指定公司，年份的表格数据
                        pdf_table = load_tables_of_years(company, years, pdf_tables, pdf_info)
                    # 既不是增长率也不是公式计算
                    else: 
                        logger.error('无法匹配, 该问题既不是增长率也不是公式计算')
                        pdf_table = load_tables_of_years(company, years, pdf_tables, pdf_info)
                    # 获取分步问题，分步年份，分步关键词，公式，完整问题公式  获取step_questions（分步问题列表） step_keywords（分步关键词列表）variable_names（变量名称列表）、step_years（分步年份列表）、formula（公式）、question_formula（问题公式）
                    step_questions, step_keywords, variable_names, step_years, formula, question_formula = type2.get_step_questions(
                        ori_question, ''.join(question_keywords), real_comp, years[0])
                    # 初始化分步的答案
                    step_answers = []
                    variable_values = []
                    if len(step_questions) > 0:
                        # 遍历每个步骤
                        for step_question, step_keyword, step_year in zip(step_questions, step_keywords, step_years):
                            if len(step_keyword) == 0:
                                logger.error('关键词为空')

                            background = '已知{}{}年的资料如下:\n'.format(real_comp, step_year)
                            # background += '----------------------------------------\n'
                            # 用于从PDF表格数据中召回与给定关键词匹配的行数据
                            matched_table_rows = recall_pdf_tables(step_keyword, [step_year], pdf_table, 
                                min_match_number=3, top_k=5)
                            # print(matched_table_rows)
                            if len(matched_table_rows) == 0:
                                logger.warning('无法匹配keyword {}, 尝试不设置限制'.format(step_keyword))
                                matched_table_rows = recall_pdf_tables(step_keyword, [step_year], pdf_table, 
                                min_match_number=2, top_k=None)
                            if len(matched_table_rows) == 0:
                                logger.error('仍然无法匹配keyword {}'.format(step_keyword))
                                matched_table_rows = recall_pdf_tables(step_keyword, [step_year], pdf_table, 
                                min_match_number=0, top_k=10)
                            # 将匹配的表格行转换为自然语言描述
                            table_text = table_to_text(real_comp, ori_question, matched_table_rows, with_year=False)
                            if table_text != '':
                                # 通过分步查询到表格信息转化为自然语言后添加到资料中。
                                background += table_text  
                            # 根据查询到的资料和问题给出规范的提示词
                            question_for_model = prompt_util.get_prompt_single_question(ori_question, real_comp, step_year).format(background, step_question)
                            logger.opt(colors=True).info('<cyan>{}</>'.format(question_for_model.replace('<', '')))
                            # 模型输出该步骤的答案
                            step_answer = model(question_for_model)
                            # 从输出答案中提取有效数值
                            variable_value = type2.get_variable_value_from_answer(step_answer)
                            if variable_value is not None:
                                # 保存该步骤的答案
                                step_answers.append(step_answer)
                                variable_values.append(variable_value)
                            logger.opt(colors=True).info('<green>{}</><red>{}</>'.format(step_answer.replace('<', ''), variable_value))
                    if len(step_questions) == len(variable_values):
                        # 将分步提取的变量值代入公式进行计算
                        for name, value in zip(variable_names, variable_values):
                            formula = formula.replace(name, value)
                        result = None
                        # 公式计算
                        try:
                            result = eval(formula)
                        except:
                            logger.error('Eval formula {} failed'.format(formula))
                        # 结果处理
                        if result is not None:
                            answer = ''.join(step_answers)
                            answer += question_formula
                            answer += '得出结果{:.2f}({:.2f}%)'.format(result, result*100)
                            logger.opt(colors=True).info('<magenta>{}</>'.format(answer.replace('<', '')))

            elif question_type == 'E':
                logger.info('这是个统计题')
                # 构建SQL文件路径
                sql_csv = os.path.join(cfg.DATA_PATH, 'sql', '{}.csv'.format(question['id']))
                if os.path.exists(sql_csv):
                    with open(sql_csv, 'r', encoding='utf-8') as f:
                        sql_result = json.load(f)
                        sql = sql_result['sql']
                    if sql is not None:
                        # 替换SQL中的特定关键词
                        sql = sql.replace('总资产', '资产总计')
                        sql = sql.replace('总负债', '负债合计')
                        sql = sql.replace('资产总额', '资产总计')
                        sql = sql.replace('其余资产', '其他流动资产')
                        sql = sql.replace('公司注册地址', '注册地址')
                        # 修正SQL查询中的数字格式，确保与问题中的数字一致
                        sql = sql_correct_util.correct_sql_nuxmber(sql, ori_question)
                        # 执行SQL查询，获取答案和执行日志。
                        answer, exec_log = sql_correct_util.exc_sql(ori_question, sql, sql_cursor)

                        if answer is None: # 识别字段错误
                            # sql错误尝试修复一次
                            try:
                                if 'no such column' in exec_log:
                                    # 对不正确的字段进行修改矫正
                                    sql = sql_correct_util.correct_sql_field(sql, ori_question, model)
                                    # 重试执行
                                    answer, _ = sql_correct_util.exc_sql(ori_question, sql, sql_cursor)
                                else:
                                    # 记录原始SQL（脱敏处理）
                                    logger.info('模型纠正前sql：{}'.format(sql.replace('<>', '')))
                                    # 调用模型生成修正SQL
                                    correct_sql_answer = model(prompt_util.prompt_sql_correct.format(key_words, sql, str(e)))
                                    logger.info('模型纠正sql结果：{}'.format(correct_sql_answer.replace('<>', '')))
                                    sql_result = re.findall('```sql([\s\S]+)```', correct_sql_answer)
                                    if len(sql_result) > 0:
                                        sql = sql_result[0].replace('\n','').strip()
                                    # 执行修正后的SQL
                                    logger.info('模型纠正后sql：{}'.format(sql.replace('<>', '')))  # 记录新SQL
                                    answer, _ = sql_correct_util.exc_sql(ori_question, sql, sql_cursor) # 重试执行
                            except Exception as e:
                                logger.error('纠正SQL[{}]错误! {}'.format(sql.replace('<>', ''), e))

                        logger.opt(colors=True).info('<green>{}</>'.format(sql.replace('<>', '')))
                        logger.opt(colors=True).info('<magenta>{}</>'.format(str(answer).replace('<>', '')))
            # 识别问题类型为开放类题
            elif question_type == 'F':  # 处理类型F的问题
                if len(years) == 0:   # 没有检测到年份
                    logger.warning('匹配到Type3-2') # 记录问题分类日志
                    answer = model(ori_question) # 直接调用模型回答原始问题
                    logger.opt(colors=True).info('<magenta>{}</>'.format(answer.replace('<', '')))
                elif len(company_abbrs) == 0:  # 检测到年份但未匹配公司简称
                    logger.warning('问题存在年份, 但没有匹配的年报')  # 提示缺少必要数据
                else: # 年份和公司均有效
                    # 解析问题关键词
                    anoy_question, _ = question_util.parse_question_keywords(model, ori_question, real_comp, years)
                    logger.info('问题关键词: {}'.format(question_keywords))  # 记录提取的关键词
                    # 加载对应年份的PDF表格数据
                    pdf_table = load_tables_of_years(company, years, pdf_tables, pdf_info)
                    # 构建背景信息头
                    background = '***************{}{}年年报***************\n'.format(
                        real_comp, years[0])
                    # 通过BM25 根据问题两个方面进行召回相关年报文本片段
                    matched_text = recall_annual_report_texts(model, anoy_question, ''.join(question_keywords), 
                        mactched_pdf_names[0], None)
                    # 组装文本片段到背景信息
                    for block_idx, text_block in enumerate(matched_text):
                        background += '{}片段:{}{}\n'.format('-'*15, block_idx+1, '-'*15)
                        background += text_block
                        background += '\n'
                    # 构建模型输入Prompt
                    question_for_model = prompt_util.prompt_question_tp31.format(
                        background, ori_question, ''.join(question_keywords),
                        ''.join(question_keywords), ''.join(question_keywords))
                    # 处理输入长度限制
                    logger.info('Prompt length {}'.format(len(question_for_model)))
                    if len(question_for_model) > 5120:
                        question_for_model = question_for_model[:5120]
                    # 执行模型调用
                    logger.info(question_for_model.replace('<', ''))
                    answer = model(question_for_model)
                    logger.info('Answer length {}'.format(len(answer)))
                    logger.opt(colors=True).info('<magenta>{}</>'.format(answer.replace('<', '')))
        except Exception as e:
            print(e)

        result = copy.deepcopy(question)
        if answer is not None:
            result['answer'] = answer
        else:
            logger.error('问题无法找到类别, 无法回答')
            result['answer'] = ''
        
        with open(answer_csv, 'w', encoding='utf-8') as f:
            try:
                json.dump(result, f, ensure_ascii=False)
            except:
                result['answer'] = ''
                json.dump(result, f, ensure_ascii=False)

'''
该函数主要用于：加载测试问题集 检查预生成答案缓存 执行答案后处理（如脱敏）
'''
def make_answer():
    answers = [] # 初始化答案存储列表
    # 加载测试问题集（假设返回结构：[{'id':1, 'question':'...'}, ...]）
    test_questions = load_test_questions()
    # 构建答案存储目录路径（如：./data/answers）
    answer_dir = os.path.join(cfg.DATA_PATH, 'answers')
    # 遍历每个测试问题
    for question in test_questions:
        # 构造答案文件路径（如：./data/answers/1.csv）
        answer_csv = os.path.join(answer_dir, '{}.csv'.format(question['id']))

        # 检查是否已有缓存答案
        if os.path.exists(answer_csv):
            # 读取已存在的答案文件
            with open(answer_csv, 'r', encoding='utf-8') as f:
                answer = json.load(f)
                question = answer
        else: # 无缓存则初始化空答案
            question['answer'] = ''
        # 答案重写处理（如去除敏感信息）
        question['answer'] = re_util.rewrite_answer(question['answer'])
        # 收集处理后的答案
        answers.append(question)
    # 构建输出路径（如：./data/result_20231025.json）
    save_path = os.path.join(cfg.DATA_PATH, 'result_{}.json'.format(datetime.now().strftime('%Y%m%d')))
    # 写入最终结果文件
    with open(save_path, 'w', encoding='utf-8') as f:
        for answer in answers:
            try:
                line = json.dumps(answer, ensure_ascii=False).encode('utf-8').decode() + '\n'
            except:
                answer['answer'] = ''
                line = json.dumps(answer, ensure_ascii=False).encode('utf-8').decode() + '\n'
            f.write(line)
