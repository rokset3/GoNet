from typing import Optional

import pandas as pd
import numpy as np



def extract_latency_features_from_df(dataframe: pd.DataFrame,
                                     PT_col_name: 'str',
                                     RT_col_name: 'str',
                                     keycode_col_name: 'str',
                                     participant_id_col_name: 'str',
                                     test_section_id_col_name: 'str')-> dict:
    '''
    This function takes dataframe with columns 
    PRESS_TIME -> press time in UNIX time
    RELEASE_TIME -> release time in UNIX time
    KEYCODE -> code of pressed key
    Args:
        dataframe: pd.DataFrame         - dataframe with columns of interest
        PT_col_name: 'str'              - name of PRESS_TIME column
        RT_col_name: 'str'              - name of RELEASE_TIME column
        keycode_col_name: 'str'         - name of keycode column
        participant_id_col_name: 'str'  - name of column with participant id
        test_section_id_col_name: 'str' - name of column with test_section id  
    Returns:
        dictionary with extracted features
    '''
    assert len(dataframe[f'{participant_id_col_name}'].unique().tolist()) == 1, f'Expected data from one unique participant'
    assert len(dataframe[f'{test_section_id_col_name}'].unique().tolist()) == 1, f'Expected data from one unique test sample'
    
    participant_id = dataframe[f'{participant_id_col_name}'].unique()[0]
    section_id = dataframe[f'{test_section_id_col_name}'].unique()[0]

        
    dataframe = dataframe.copy()

    for col in [PT_col_name, RT_col_name]:
        dataframe[f'{col}_lag'] = dataframe[f'{col}'].shift(1)
        dataframe[f'{col}_lag'] = np.where(dataframe[f'{col}_lag'].isna(), dataframe[f'{col}'], dataframe[f'{col}_lag'])
        dataframe[f'{col}_lag'] = dataframe[f'{col}_lag'].astype('int64')


    dataframe['HL'] = dataframe[f'{RT_col_name}'] - dataframe[f'{PT_col_name}']
    dataframe['IL'] = dataframe[f'{PT_col_name}'] - dataframe[f'{RT_col_name}_lag']
    dataframe['PL'] = dataframe[f'{PT_col_name}'] - dataframe[f'{PT_col_name}_lag']
    dataframe['RL'] = dataframe[f'{RT_col_name}'] - dataframe[f'{RT_col_name}_lag']

    hl = dataframe["HL"].tolist()
    il = dataframe['IL'].tolist()
    pl = dataframe['PL'].tolist()
    rl = dataframe['RL'].tolist()
    output_dict = dict(
        participant_id = participant_id,
        section_id = section_id,
        keycode_ids = dataframe[f'{keycode_col_name}'].tolist(),
        hl = hl,
        il = il,
        pl = pl,
        rl = rl 
        )


    return output_dict
    