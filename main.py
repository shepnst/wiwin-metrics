import json
from func_to_call import parse_all_data, parse_data_with_time
from metrics import Validator
import time
from queue import Queue
import threading
import datetime

lock = threading.Lock()
event = threading.Event()
json_data_queue = Queue()
list_data_queue = Queue(maxsize=50)
number_of_logs = 0
number_of_processed_json_logs = 0
number_of_processed_list_data = 0

data_v1 = parse_all_data('datasets/val_set.json')

data_v2 = parse_data_with_time('datasets/val_set.json')

with open('parsed_tuning.json', 'w', encoding='utf-8') as f:
    json.dump(data_v1, f, ensure_ascii=False)

with open('parsed_dash.json', 'w', encoding='utf-8') as f:
    json.dump(data_v2, f, ensure_ascii=False)

with open('parsed_dash.json', 'r', encoding='utf-8') as f:
    formatted_data = json.load(f)

def producer():
    global lock
    global json_data_queue
    global list_data_queue
    global number_of_processed_json_logs
    while True:
        with lock:
            if json_data_queue.empty():
                break
            else:
                try:
                    new_data = json_data_queue.get()
                    while list_data_queue.full():  # Проверка на заполненность
                        lock.release()  # Освобождаем блокировку, чтобы другие потоки могли работать
                        time.sleep(0.1)  # Ждём одну секунду, чтобы проверить на заполненность снова
                        lock.acquire()  # Снова захватывается блокировка
                    list_data_queue.put(new_data)
                finally:  # гарантируем выполнение этого
                    number_of_processed_json_logs += 1

def consumer():
    global lock
    global event
    global list_data_queue
    global number_of_logs
    global number_of_processed_json_logs
    global number_of_processed_list_data
    # Consumer работает пока кол-во обработанных путей < кол-ва переданных путей
    while True:
        # Выполняется цикл while, пока из очереди не удастся извлечь изображение
        while True:
            with lock:
                if list_data_queue.empty():
                    if number_of_processed_json_logs == number_of_logs:
                        return
                    continue
                else:
                    new_data = list_data_queue.get()
                    break

        with lock:
            vs = Validator(neural=False)
            new_scores = vs.validate_rag(new_data)
            print(new_scores)
            number_of_processed_list_data += 1
            if number_of_processed_list_data == number_of_logs:
                event.set()


def one_thread(arr):
    for i in range(len(arr)):
        new_data = arr[i]
        vs = Validator(neural=False)
        new_scores = vs.validate_rag(new_data)
        print(new_scores)
    event.set()


if __name__ == "__main__":

    print ("Please enter the number of threads: ", end = "")
    num_thread = input()

    try:
        val = int(num_thread)
    except ValueError:
        print("That is not an int!")
        exit()
    if int(num_thread) < 1:
        print("The number of threads should be more than 0!")
        exit()

    dataset = [] # массив данных (для случая, когда num_threads = 1)

    number_of_logs = 0
    for item in formatted_data:
        if number_of_logs == 125:
            continue
        satisfaction = "yes"
        if 'refined_question' in item.keys():
            satisfaction = "no"
        one_log = {'question': item['user_question'],
                   'answer': item['saiga_answer'],
                   'ground_truth': item['giga_answer'],
                   'contexts': item['contexts'],
                   'satisfaction': satisfaction,
                   'time spent': item['response_time']}

        if int(num_thread) == 1:
            dataset.append(one_log)
        else:
            json_data_queue.put(one_log)
        number_of_logs += 1

    start = datetime.datetime.now()

    if int(num_thread) == 1:
        one_thread(dataset)

    else:
        with lock:
            for n in range(int(num_thread)):
                if n % 3 == 1:
                    t = threading.Thread(target=producer)
                    t.start()
                else:
                    t = threading.Thread(target=consumer)
                    t.start()

    event.wait()
    finish = datetime.datetime.now()
    print('Время работы: ' + str(finish - start))