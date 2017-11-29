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
    questions = _convert_to_dictionary(questions)

    teacher_answers_list = []
    student_answers_list = []
    questions_list = []
    scores_list = []
    keys_list = []
    for key in student_answers.id.unique().tolist():
        masked_student_answers = student_answers[student_answers.id == key]
        masked_scores = scores[key]
        for score_set in masked_scores:
            for mask_arg in range(len(masked_student_answers.text)):
                student_answer = masked_student_answers.text.iloc[mask_arg]
                student_answers_list.append(student_answer)
                scores_list.append(score_set[mask_arg])
                teacher_answers_list.append(teacher_answers[key])
                questions_list.append(questions[key])
                keys_list.append(key)

    return pd.DataFrame({'id': keys_list,
                         'questions': questions_list,
                         'teacher': teacher_answers_list,
                         'student': student_answers_list,
                         'scores': scores_list})


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


data = load_data()
