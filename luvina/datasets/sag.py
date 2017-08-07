import pandas as pd


def get_data():
    data_path = '../datasets/short_answer_grading_v2/raw/'
    questions_filename = data_path + 'questions'
    teacher_answers_filename = data_path + 'answers'
    student_answers_filename = data_path + 'all'
    questions = read_file(questions_filename)
    teacher_answers = read_file(teacher_answers_filename)
    student_answers = read_file(student_answers_filename)
    return questions, teacher_answers, student_answers


def read_file(filename):
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


questions, teacher_answers, student_answers = get_data()
