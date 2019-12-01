# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:43:13 2019

@author: Zhou
"""

import os
import re
import torch
import numpy as np
from torch import optim
import datetime
import logging
import itertools

from model import LSTMTagger, TwoLayerLSTMTagger
from config import NULL_TAG, INS_TAG, EOS_TAG
from Encoders import XEncoder, YEncoder
#from data_load import XYDataLoader, HtmlDataLoader
import lg_utils
from data_save import ExcelSaver, HtmlSaver


    

if __name__ == "__main__":
    #TODO: argparse
    # I/O settings
    # TODO: put into config
    OUTPUT_PATH = os.path.join(os.getcwd(), "result")
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    PAGE_MODEL_PATH = os.path.join(MODEL_PATH, "page_model")
    RECORD_MODEL_PATH = os.path.join(MODEL_PATH, "record_model")
    EMBEDDING_PATH = os.path.join(os.getcwd(), "Embedding", "polyglot-zh_char.pkl")
#    SOURCE_TYPE = "XY"
    DATASIZE = "full"
    SAVER_TYPE = "html"
        
    # Training settings
#    N_SECTION_TRAIN = 1#30
#    N_SECTION_TEST = 1
#    CV_PERC = 0.5
    
    N_EPOCH_PAGE = 20
    N_CHECKPOINT_PAGE = 1
    N_SAVE_PAGE = 5
    LEARNING_RATE_PAGE = 0.3
    HIDDEN_DIM_PAGE = 6
    N_EPOCH_RECORD = 20
    N_CHECKPOINT_RECORD = 1
    N_SAVE_RECORD = 5
    LEARNING_RATE_RECORD = 0.5
    HIDDEN_DIM_RECORD = 8
    
    NEED_TRAIN_MODEL = False
    USE_REGEX = False
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Logging
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=os.path.join("log",
                                              "run{}.log".format(curr_time)),
                        format='%(asctime)s %(message)s', 
                        filemode='w') 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info("Started training at {}".format(curr_time))
    
    
    pages_train = lg_utils.load_data_from_pickle("pages_train.p", DATASIZE)
    pages_cv = lg_utils.load_data_from_pickle("pages_cv.p", DATASIZE)
    pages_test = lg_utils.load_data_from_pickle("pages_test.p", DATASIZE)
    records_train = lg_utils.load_data_from_pickle("records_train.p", DATASIZE)
    records_cv = lg_utils.load_data_from_pickle("records_cv.p", DATASIZE)
    records_test = lg_utils.load_data_from_pickle("records_test.p", DATASIZE)
    
    # Set up data loaders
#    if SOURCE_TYPE == "XY":
#        loader = XYDataLoader()
#    elif SOURCE_TYPE == "html":
#        loader = HtmlDataLoader()
#    else:
#        raise ValueError
#    test_loader = XYDataLoader()
    
    # Model hyper-parameter definition
    EMBEDDING_DIM = 64          # depending on pre-trained word embedding model
    char_encoder = XEncoder(EMBEDDING_DIM, EMBEDDING_PATH)
#    interested_tags = [loader.get_person_tag()]
#    if SOURCE_TYPE == "XY":
#        interested_tags.extend(["任職時間"])
##    interested_tags = ["人名", "任職時間", "籍貫", "入仕方法"]
#    elif SOURCE_TYPE == "html":
#        interested_tags.extend(["post_time", "office", "jiguan"])
        
    page_tag_encoder = YEncoder([INS_TAG, EOS_TAG])
    tagset = set(itertools.chain.from_iterable([r.orig_tags for r in records_train]))
    record_tag_encoder = YEncoder([NULL_TAG, "<BEG>", "<END>"] + list(tagset))
    print([NULL_TAG, "<BEG>", "<END>"] + list(tagset))
    
    
    raise RuntimeError
    page_model = LSTMTagger(logger, EMBEDDING_DIM, HIDDEN_DIM_RECORD, 
                              record_tag_encoder.get_tag_dim(), bidirectional=False)
#    page_model = TwoLayerLSTMTagger(logger, EMBEDDING_DIM, HIDDEN_DIM_PAGE,
#                                    page_tag_encoder.get_tag_dim(), bidirectional=True)
    record_model = LSTMTagger(logger, EMBEDDING_DIM, HIDDEN_DIM_RECORD, 
                              record_tag_encoder.get_tag_dim(), bidirectional=True)
    page_optimizer = optim.SGD(page_model.parameters(), lr=LEARNING_RATE_PAGE)
    record_optimizer = optim.SGD(record_model.parameters(), lr=LEARNING_RATE_RECORD)
    
    # Load training, CV and testing data
#    pages_train, pages_cv, records_train, records_cv = loader.load_data(interested_tags,
#                                                                    "train",
#                                                                    N_SECTION_TRAIN,
#                                                                    cv_perc=CV_PERC)
#    pages_test, _ = test_loader.load_data(interested_tags, 
#                                            "test",
#                                            N_SECTION_TEST)
    
    # Load models if it was previously saved and want to continue
    if os.path.exists(PAGE_MODEL_PATH) and not NEED_TRAIN_MODEL:
        page_model.load_state_dict(torch.load(os.path.join(PAGE_MODEL_PATH, "final.pt")))
        page_model.eval()
    if os.path.exists(RECORD_MODEL_PATH) and not NEED_TRAIN_MODEL:
        record_model.load_state_dict(torch.load(os.path.join(RECORD_MODEL_PATH, "final.pt")))
        record_model.eval()
    
    # Training
    # Step 1. Data preparation
    page_training_data = lg_utils.get_data_from_samples(pages_train,
                                                        char_encoder,
                                                        page_tag_encoder)
    page_cv_data = lg_utils.get_data_from_samples(pages_cv,
                                                  char_encoder,
                                                  page_tag_encoder)
    page_test_data = [p.get_x(char_encoder) for p in pages_test]
    
    record_training_data = lg_utils.get_data_from_samples(records_train,
                                                          char_encoder,
                                                          record_tag_encoder)
    record_cv_data = lg_utils.get_data_from_samples(records_cv,
                                                    char_encoder,
                                                    record_tag_encoder)
    
    # Step 2. Model training
    if NEED_TRAIN_MODEL:
        # 2.a Train model to parse pages into sentences
        if not USE_REGEX:
            page_model.train_model(page_training_data, page_cv_data, 
                                   page_optimizer, "NLL",
                                   N_EPOCH_PAGE, N_CHECKPOINT_PAGE,
                                   N_SAVE_PAGE, PAGE_MODEL_PATH)
        # 2.b Train model to tag sentences
        record_model.train_model(record_training_data, record_cv_data, 
                                 record_optimizer, "NLL",
                                 N_EPOCH_RECORD, N_CHECKPOINT_RECORD,
                                 N_SAVE_RECORD, RECORD_MODEL_PATH)
        
    # Evaluate on test set
    # Step 1. using page_to_sent_model, parse pages to sentences
    if USE_REGEX:
        with open(os.path.join(MODEL_PATH, "surname.txt"), 'r', encoding="utf8") as f:
            surnames = f.readline().replace("\ufeff", '')
        tag_seq_list = []
        for p in pages_test:
            tags = [INS_TAG for c in p.txt]
            for m in re.finditer(r"○("+surnames+')', p.txt):
                tags[m.start(0)] = EOS_TAG  # no need to -1, instead drop '○' before name
            tags[-1] = EOS_TAG
            tag_seq_list.append(tags)
    else:
        tag_seq_list = page_model.evaluate_model(page_test_data, page_tag_encoder)
    record_test_data = []
    records = []
    for p, pl in zip(pages_test, lg_utils.get_sent_len_for_pages(tag_seq_list, EOS_TAG)):
        rs = p.separate_sentence(pl)
        records.extend(rs)
        record_test_data.extend([r.get_x(char_encoder) for r in rs])
            
    # Step 2. using sent_to_tag_model, tag each sentence
    tagged_sent = record_model.evaluate_model(record_test_data, record_tag_encoder)
    for record, tag_list in zip(records, tagged_sent):
        record.set_tag(tag_list)
    
    # Calculate the error rate on training set
#    inputs = [r.get_x(char_encoder) for r in records_train]
#    tag_pred = record_model.evaluate_model(inputs, record_tag_encoder)
#    tag_true = [[c.get_tag() for c in r.chars] for r in records_train]
    correct_ratio_train = lg_utils.correct_ratio_calculation(records_train, 
                                char_encoder, record_model, record_tag_encoder)
    print("The correct ratio of train set is {}".format(correct_ratio_train))
    
    # Calculate the error rate on cv set
    corrcect_ratio_cv = lg_utils.correct_ratio_calculation(records_cv, 
                                char_encoder, record_model, record_tag_encoder)
    
    print("The correct ratio of cv set is {}".format(corrcect_ratio_cv))
    
    raise RuntimeError
        
    # Saving
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if SAVER_TYPE == "html":
        saver = HtmlSaver(records)
        filename = os.path.join(OUTPUT_PATH, "test_{}.txt".format(curr_time))
    else:
        saver = ExcelSaver(records)
        filename = os.path.join(OUTPUT_PATH, "test_{}.xlsx".format(curr_time))
    saver.save(filename, interested_tags)
