from luvina.utils.data_utils import get_file
import pandas as pd
import glob
from os.path import basename
import os


def _get_string_data(root_path):
    # data_path = '../datasets/short_answer_grading_v2/raw/'
    data_path = root_path + '/raw/'
    questions_filename = data_path + 'questions'
    teacher_answers_filename = data_path + 'answers'
    student_answers_filename = data_path + 'all'
    questions = _read_file(questions_filename)
    teacher_answers = _read_file(teacher_answers_filename)
    student_answers = _read_file(student_answers_filename)
    return questions, teacher_answers, student_answers


def _read_file(filename):
    data_file = open(filename, 'r', encoding='latin-1')
    texts = []
    data_ids = []
    for line in data_file:
        if line[3] == ' ':
            data_id = line[:3]
            text = line[4:-2]
        elif line[4] == ' ':
            data_id = line[:4]
            text = line[5:-2]
        elif line[5] == ' ':
            data_id = line[:5]
            text = line[6:-2]
        else:
            raise NotImplementedError('Space value not found')
        data_ids.append(data_id)
        texts.append(text)
    data_frame = pd.DataFrame({'id': data_ids, 'text': texts})
    return data_frame


def _get_scores(root_path):
    # data_path = '../datasets/short_answer_grading_v2/scores/'
    data_path = root_path + '/scores/'
    score_paths = glob.glob(data_path + '*')
    scores = dict()
    for scores_path in score_paths:
        score_files = glob.glob(scores_path + '/*')
        question_scores = []
        for score_file in score_files:
            file_scores = []
            _score_file = open(score_file, 'r')
            for line in _score_file:
                file_scores.append(float(line))
            question_scores.append(file_scores)
        scores_key = basename(scores_path)
        scores[scores_key] = question_scores
    return scores


def _convert_to_dictionary(string_data):
    data = dict()
    for arg, key in enumerate(string_data.id.tolist()):
        data[key] = string_data.text.iloc[arg]
    return data


def get_data(root_path):
    questions, teacher_answers, student_answers = _get_string_data(root_path)
    scores = _get_scores(root_path)
    teacher_answers = _convert_to_dictionary(teacher_answers)
    teacher_answers_list = []
    student_answers_list = []
    scores_1 = []
    scores_2 = []
    scores_3 = []
    keys = student_answers.id.tolist()
    for arg, key in enumerate(keys):
        teacher_answers_list.append(teacher_answers[key])
        student_answers_list.append(student_answers.text.iloc[arg])
        scores_list = scores[key]
        scores_1.append(scores_list[0])
        scores_2.append(scores_list[1])
        scores_3.append(scores_list[2])
    data_1 = pd.DataFrame({'teacher': teacher_answers_list,
                           'student': student_answers_list,
                           'score': scores_1})

    data_2 = pd.DataFrame({'teacher': teacher_answers_list,
                           'student': student_answers_list,
                           'score': scores_2})

    data_3 = pd.DataFrame({'teacher': teacher_answers_list,
                           'student': student_answers_list,
                           'score': scores_3})

    data_frames = [data_1, data_2, data_3]
    data = pd.concat(data_frames, axis=0)
    return data


def load_data(path='ShortAnswerGrading_v2.0'):
    """Loads the Short Answer grading data.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    origin = ('http://web.eecs.umich.edu/~mihalcea/' +
              'downloads/ShortAnswerGrading_v2.0.zip')
    root_path = get_file(path, origin=origin, extract=True,
                         cache_subdir='datasets/short_answer_grading')
    root_path = os.path.dirname(root_path) + '/data'
    return get_data(root_path)
