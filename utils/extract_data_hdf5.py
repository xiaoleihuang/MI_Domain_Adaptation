import glob
import h5py
import os
from os.path import basename
import json
from . import data_helper
import numpy as np


def tmp_c(tet):
    """This function is an assistant for the byte_utf8_converter
    """
    if tet == None or len(tet) == 0:
        return None
    try:
        return tet.decode('utf-8')
    except UnicodeDecodeError:
        return None


def byte_utf8_converter(bt):
    """This function is to convert byte code to line only if UnicodeDecodeError encounters.
    """
    idx = 0
    try:
        return bt.decode('utf-8')
    except UnicodeDecodeError:
        tmp = None
        while idx < len(bt) - 5 and tmp is None:
            idx += 1
            tmp = bt[idx:]
            tmp = tmp_c(tmp)
        if tmp != None:
            return tmp
        else:
            return ''


def extract_hdf5(directory, filters={'SPEAKER': set(['P'])}, min_seqs=5, max_seqs=-1, include_pre_contxt=False,
                 code_num=10):
    """Extract data from HDF5 files

    This function is to extract data from HDF5 files by the 'WORDS' Field.

    Args:
        directory (str): The directory path of HDF5 files.
        filters (dict): The data items we want to filter out from the HDF5 data.
        min_seqs (int): The minimum length of the sentence (exclusive).
        max_seqs (int): The maximum length of the sentence (exclusive).
        include_pre_contxt (bool): Whether includes the context information.
        code_num (int): latest n number of annotations, 10 is the default

    Examples:
        extract_hdf5("../MI_hdf5/*.hdf5")
    Returns:
        generator: a list of sentences & associated labels, but to be memory efficient, it will return a generator by yield
    """
    filter_flag = False
    filelist = glob.glob(directory)
    # iterate each file path in the file list
    for filep in filelist:
        data = h5py.File(filep, 'r')
        talk_len = len(data['WORDS'])
        topic_step = int(talk_len/3) + 1

        for index, line in enumerate(data['WORDS']):
            # filter dataset defined in filters
            for filter_key in filters:
                if filter_key in data.keys() and data[filter_key][index].decode('utf-8').strip() not in filters[
                    filter_key]:
                    filter_flag = True
                    break
            if filter_flag:
                filter_flag = False
                continue
            
            topic_domain = [0] * 3 # create domain label here
            topic_domain[int(index/topic_step)] = 1

            # generate labels
            """label = 0
            if data['CHANGEPROB'][index] < 0: # CHANGEBINGE
                label = 1"""
            """Create contexts from CODES"""
            code_padding = 'unknown'  # for if the code_num larger than the index
            if index < 10:
                code_list = [code_padding] * (code_num - index)
                for tmp_idx_code in range(index):
                    code_list.append(data['CODE'][tmp_idx_code].decode('utf-8'))
            else:
                #code_list = data['CODE'][index - 10:index]
                code_list = [item.decode('utf-8') for item in code_list if type(item) != str] # unicode
            # preprocess codes
            code_list_new = []
            for code_tmp in code_list:
                if '+' in code_tmp or '-' in code_tmp:
                    # if sys.version_info[0] == 2:
                    #	code_tmp_new = unicode(code_tmp[:-2]+str(data_helper.convert_code(code_tmp)))
                    # else:
                    code_tmp_new = code_tmp[:-2] + str(data_helper.convert_code(code_tmp))
                    code_list_new.append(code_tmp_new)
                else:
                    code_list_new.append(code_tmp)

            # create content
            try:
                seqs = line.decode('utf-8').strip()
            except UnicodeDecodeError:
                print(seqs)
                seqs = byte_utf8_converter(line).strip()
                # continue
                # print(seqs)
                if len(seqs) == 0:
                    continue

            # validate min and max limitations
            if min_seqs != -1 and len(seqs.split(' ')) < min_seqs:
                continue
            if max_seqs != -1 and len(seqs.split(' ')) > max_seqs:
                continue

            # add context from interviewer's questions
            if include_pre_contxt:
                tmp_idx = index - 1
                context = ""
                while tmp_idx > index - 5 and tmp_idx > -1 and data['SPEAKER'][tmp_idx] != 'I':
                    try:
                        context = data['WORDS'][tmp_idx].decode('utf-8').strip() + ' ' + data['CODE'][tmp_idx].decode(
                            'utf-8').strip() + ' ' + '<P>' + ' ' + context
                    except UnicodeDecodeError:
                        # break
                        tmp_idx -= 1
                        continue
                        # context = byte_utf8_converter(data['WORDS'][tmp_idx].decode('utf-8')) + ' ' + byte_utf8_converter(data['CODE'][tmp_idx].decode('utf-8')) + ' '  + '<P>' + ' ' + seqs
                    tmp_idx -= 1
                if tmp_idx > -1:
                    try:
                        context = data['CODE'][tmp_idx].decode('utf-8').strip() + ' ' + data['WORDS'][tmp_idx].decode(
                            'utf-8').strip() + ' ' + '<I>' + ' ' + context
                    except UnicodeDecodeError:
                        pass
                        # context = byte_utf8_converter(data['CODE'][tmp_idx].decode('utf-8')) + ' ' + byte_utf8_converter(data['WORDS'][tmp_idx].decode('utf-8')) + ' ' + '<I>' + ' '  + context

                yield (seqs, context, code_list_new, data['CODE'][index].decode('utf-8'), topic_domain)
            else:
                yield (seqs, None, code_list_new, data['CODE'][index].decode('utf-8'), topic_domain)


def extract_hdf5_10segs_people(directory, filters={'SPEAKER': set(['P'])}, min_seqs=5, max_seqs=-1, split_num=10):
    """Extract data from HDF5 files: each people will be segmented into 10 euqal parts, each part will be treated as equally
        The process should be similar to the  extract_hdf5

    Args:
        split_num (int): it defines each people will be splitted into the number of parts

    Returns:
        features for each segments (people): each segment contains a list of contents, a list of corresponding contexts, a list of corespondding last 10 codes, and a predicting target.
    """
    filter_flag = False
    filelist = glob.glob(directory)

    # iterate each file path in the file list
    for filep in filelist:
        data = h5py.File(filep, 'r')
        print(filep)
        len_num = len(data['WORDS'])
        seg_nums = np.array_split(np.arange(len_num), split_num)

        for indices in seg_nums:
            # _p refers to people's level, which contains multiple utterances
            contents_p = []
            contexts_p = []
            ctxt_codes_p = []
            # for predict target: ChangeProb (current), -1, 0, or 1
            label_p = data["CHANGEPROB"][indices[0]]
            if label_p > 0:
                label_p = 1
            elif label_p < 0:
                label_p = -1
            else:
                label_p = 0
            label_p = str(label_p)  # encode the label

            for index in indices:
                # filters for contents
                for filter_key in filters:
                    if filter_key in data.keys() and data[filter_key][index].decode('utf-8').strip() not in filters[
                        filter_key]:
                        filter_flag = True
                        break
                if filter_flag:
                    filter_flag = False
                    continue

                """contents"""
                try:
                    content = data['WORDS'][index].strip().decode('utf-8').strip()
                except UnicodeDecodeError:
                    continue

                # control the length
                if min_seqs > 0 and len(content.split()) > min_seqs:
                    continue
                if max_seqs > 0 and len(content.split()) < max_seqs:
                    continue

                contents_p.append(content)

                """codes"""
                code_list = []
                # for if the code_num larger than the index
                code_padding = 'unknown'
                tmp_index = indices[0]  # the start point is the 1st code in this segment
                if index < 10 + tmp_index:
                    code_list = [code_padding] * (10 - index + tmp_index)
                    for tmp_idx_code in range(tmp_index, index):
                        code_list.append(data['CODE'][tmp_idx_code])
                else:
                    code_list = data['CODE'][index - 10:index]
                code_list_new = []
                for code_tmp in code_list:
                    if '+' in code_tmp or '-' in code_tmp:
                        code_tmp_new = code_tmp[:-2] + str(data_helper.convert_code(code_tmp))
                        code_list_new.append(code_tmp_new)
                    else:
                        code_list_new.append(code_tmp)

                ctxt_codes_p.append(" ".join(code_list_new))

                """contexts: add the last 5 same as the previous settings"""
                start_idx = index - 1
                context = ""
                while start_idx > tmp_index - 1 and start_idx > index - 5 and data['SPEAKER'][start_idx] != 'I':
                    try:
                        context = data['WORDS'][start_idx].decode('utf-8').strip() + ' ' + '<P>' + ' ' + context
                    except UnicodeDecodeError:
                        start_idx -= 1
                        continue
                    start_idx -= 1
                if start_idx > tmp_index - 1:
                    try:
                        context = data['WORDS'][start_idx].decode('utf-8').strip() + ' ' + '<I>' + ' ' + context
                    except UnicodeDecodeError:
                        pass
                contexts_p.append(context)
            yield contents_p, contexts_p, ctxt_codes_p, label_p


def extract_hdf5_by_people(directory, filter_fields={'SPEAKER': set(['P'])}, min_seqs=5, max_seqs=-1):
    """Extract data from HDF5 files

    This function is to extract data from HDF5 files by the 'WORDS' Field.

    Args:
        directory (str): The directory path of HDF5 files.
        filter_fields (dict): The data items we want to filter out from the HDF5 data fields.
        min_seqs (int): The minimum length of the sentence (exclusive).
        max_seqs (int): The maximum length of the sentence (exclusive).

    Examples:
        extract_hdf5("../MI_hdf5/*.hdf5")
    Returns:
        generator: a list of sentences & associated labels, but to be memory efficient, it will return a generator by yield
    """
    filter_flag = False

    info_data = dict()  # data for one people

    data = h5py.File(directory, 'r')  # load data

    info_data['id'] = os.path.splitext(basename(directory))[0].strip()
    # generate changebinge labels
    info_data['changeprob'] = float(data['CHANGEPROB'][0])
    info_data['changebinge'] = float(data['CHANGEBINGE'][0])

    info_data['words'] = list()  # save each sentence and its code as a tuple

    for index, line in enumerate(data['WORDS']):
        # filter dataset defined in filters
        for filter_key in filter_fields:
            if filter_key in data.keys() and data[filter_key][index].decode('utf-8') not in filter_fields[filter_key]:
                filter_flag = True
                break
        if filter_flag:
            filter_flag = False
            continue

        try:
            seqs = line.decode('utf-8')
        except UnicodeDecodeError:
            print(line)
            continue

        if min_seqs != -1 and len(seqs.split(' ')) < min_seqs:
            continue

        if max_seqs != -1 and len(seqs.split(' ')) > max_seqs:
            continue

        info_data['words'].append((seqs, data['CODE'][index].decode('utf-8')))
    return info_data


def save_dataset(data_dict, save_path=''):
    """Save the processed dataset as json for the future reference
    """
    with open(save_path, 'wb') as writefile:
        writefile.write(
            json.dumps(data_dict, ensure_ascii=False, sort_keys=True, indent=4
                       ).encode('utf-8', 'replace'))


def load_dataset(data_path):
    """Load existing json format dataset
    """
    datafile = open(data_path)
    return json.load(datafile)


if __name__ == '__main__':
    print(extract_hdf5_by_people("../MI_hdf5/*.hdf5"))
