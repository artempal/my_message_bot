# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np 
import sys
from random import randint
import datetime
from sklearn.utils import shuffle
import pickle
import os
from transliterate import translit, get_available_language_codes
import re  

reload(sys)  
sys.setdefaultencoding('utf-8')
REG_FILTER = re.compile(u'[^a-zA-Z ]') # фильтр сообщений - по умолчанию только русские символы и пробелы

def clean_message(message_body):
	message_body = translit(message_body,'ru',reversed=True) #делаем транслитерацию
	message_body = REG_FILTER.sub(' ',message_body) # фильтруем сообщения
	message_body = message_body.strip() # удалим лишние пробелы в начале и конце сообщения
	message_body = re.sub(r'\s+', ' ', message_body) # удалим лишние пробелы внутри
	return message_body


def createTrainingMatrices(conversationFileName, wList, maxLen):
	# Загружаем наш словарь
	conversationDictionary = np.load(conversationFileName).item()
	numExamples = len(conversationDictionary) 
	# Инициализируем нулями массивы обучения
	xTrain = np.zeros((numExamples, maxLen), dtype='int32')
	yTrain = np.zeros((numExamples, maxLen), dtype='int32')
	for index,(key,value) in enumerate(conversationDictionary.iteritems()):
		# Инициализируем целочисленные представления строк
		encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
		decoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
		# Получаем только отдельные слова в строке и считаем их количество
		keySplit = key.split()
		valueSplit = value.split()
		keyCount = len(keySplit)
		valueCount = len(valueSplit)
		# Пропускаем слишком длинные фразы
		if (keyCount > (maxLen - 1) or valueCount > (maxLen - 1)):
			continue
		
		for keyIndex, word in enumerate(keySplit):
			try:
				encoderMessage[keyIndex] = wList.index(word)
			except ValueError:
				encoderMessage[keyIndex] = 0

		encoderMessage[keyIndex + 1] = wList.index('<EOS>') # добавим символ конца

		for valueIndex, word in enumerate(valueSplit):
			try:
				decoderMessage[valueIndex] = wList.index(word)
			except ValueError:
				decoderMessage[valueIndex] = 0

		decoderMessage[valueIndex + 1] = wList.index('<EOS>') # добавим символ конца

		xTrain[index] = encoderMessage
		yTrain[index] = decoderMessage

	# Удаляем нулевые строки
	yTrain = yTrain[~np.all(yTrain == 0, axis=1)]
	xTrain = xTrain[~np.all(xTrain == 0, axis=1)]

	# Считаем итоговое количество тренеровочных примеров
	numExamples = xTrain.shape[0]
	return numExamples, xTrain, yTrain

def getTrainingBatch(localXTrain, localYTrain, localBatchSize, maxLen):
	num = randint(0,2)
	arr = localXTrain[num:num + localBatchSize]
	labels = localYTrain[num:num + localBatchSize]
	# Реверсируем порядок строки энкодера
	reversedList = list(arr)
	for index,example in enumerate(reversedList):
		reversedList[index] = list(reversed(example))

	# Находим специальные метки обучения
	laggedLabels = []
	EOStokenIndex = wordList.index('<EOS>')
	padTokenIndex = wordList.index('<pad>')
	for example in labels:
		eosFound = np.argwhere(example==EOStokenIndex)[0]
		shiftedExample = np.roll(example,1)
		shiftedExample[0] = EOStokenIndex
		# Токен EOS уже был в конце, поэтому не нужен PAD
		if (eosFound != (maxLen - 1)):
			shiftedExample[eosFound+1] = padTokenIndex
		laggedLabels.append(shiftedExample)

	reversedList = np.asarray(reversedList).T.tolist()
	labels = labels.T.tolist()
	laggedLabels = np.asarray(laggedLabels).T.tolist()
	return reversedList, labels, laggedLabels

def getTestInput(inputMessage, wList, maxLen):
	encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
	inputSplit = inputMessage.lower().split()
	for index,word in enumerate(inputSplit):
		try:
			encoderMessage[index] = wList.index(word)
		except ValueError:
			continue
	encoderMessage[index + 1] = wList.index('<EOS>')
	encoderMessage = encoderMessage[::-1]
	encoderMessageList=[]
	for num in encoderMessage:
		encoderMessageList.append([num])
	return encoderMessageList

# Функция перевода обратно в строку
def idsToSentence(ids, wList):
    EOStokenIndex = wList.index('<EOS>')
    padTokenIndex = wList.index('<pad>')
    myStr = ""
    listOfResponses=[]
    for num in ids:
        if (num[0] == EOStokenIndex or num[0] == padTokenIndex):
            listOfResponses.append(myStr)
            myStr = ""
        else:
            myStr = myStr + wList[num[0]] + " "
        myStr = translit(myStr,'ru') # Обратная транслитерация
    if myStr:
		listOfResponses.append(myStr)

    listOfResponses = [i for i in listOfResponses if i]
    return listOfResponses
	
def get_test_string():
	result_arr = []
	print('Введите несколько фраз, которые будут задаваться боту во время обучения для тестирования')
	print('Чтобы закончить ввод слов - введите 0 и нажмите enter')
	while(1):
		input_text = raw_input('Введите вопрос без знаков препинания или 0 для окончания ввода: ')
		if(input_text == '0' and len(result_arr) > 1):
			break
		elif(input_text == '0' and len(result_arr) <= 1):
			print('Введите хотя бы две фразы для тестирования!')
		else:
			input_text = clean_message(input_text)
			if input_text:
				result_arr.append(input_text)
			else:
				print('Строка пуста или в ней есть запрещенные символы')
	return result_arr


batchSize = 24
maxDecoderLength = maxEncoderLength = 15 # Максиммальное количество слов в сообщении
lstmUnits = 112
embeddingDim = lstmUnits
numLayersLSTM = 3
numIterations = 500000

encoderTestStrings = get_test_string() # Получаем массив строк с вопросами

# Загружаем файл со всеми словами
with open("wordList.txt", "rb") as fp:
	wordList = pickle.load(fp)

vocabSize = len(wordList) # вычисляем количество загруженных слов

# Добавим пустые символы к нашему массиву для корректного вывода
wordList.append('<pad>')
wordList.append('<EOS>')
vocabSize = vocabSize + 2

# Загружаем тренировочный словарь и создаем матрицы
numTrainingExamples, xTrain, yTrain = createTrainingMatrices('conversationDictionary.npy', wordList, maxEncoderLength)
np.save('Seq2SeqXTrain.npy', xTrain)
np.save('Seq2SeqYTrain.npy', yTrain)
print ('Завершаем загрузку словаря...')

tf.reset_default_graph()

# Создаем заполнители
encoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxEncoderLength)]
decoderLabels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
decoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
feedPrevious = tf.placeholder(tf.bool)

encoderLSTM = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits, state_is_tuple=True) # Создаем базовую сетевую яячейку типа LSTM

#encoderLSTM = tf.nn.rnn_cell.MultiRNNCell([singleCell]*numLayersLSTM, state_is_tuple=True)

# Внедряем модель seq2seq для RNN
decoderOutputs, decoderFinalState = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoderInputs, decoderInputs, encoderLSTM, 
															vocabSize, vocabSize, embeddingDim, feed_previous=feedPrevious)
# Ищем индекс с наибольшим значением по осям тензора
decoderPrediction = tf.argmax(decoderOutputs, 2)

# Оптимизируем нейронную сеть по алгоритму Адама
lossWeights = [tf.ones_like(l, dtype=tf.float32) for l in decoderLabels]
loss = tf.contrib.legacy_seq2seq.sequence_loss(decoderOutputs, decoderLabels, lossWeights, vocabSize)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Создаем сессию
sess = tf.Session()
saver = tf.train.Saver()
# Можно загрузить ранее сохраненную модель
#saver.restore(sess, tf.train.latest_checkpoint('models/'))
# Или начать новую
sess.run(tf.global_variables_initializer())

# Cоздаем папку с новым логом и загружаем граф
tf.summary.scalar('Loss', loss)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# Создаем нулевой вектор
zeroVector = np.zeros((1), dtype='int32')

for i in range(numIterations):

	encoderTrain, decoderTargetTrain, decoderInputTrain = getTrainingBatch(xTrain, yTrain, batchSize, maxEncoderLength)
	feedDict = {encoderInputs[t]: encoderTrain[t] for t in range(maxEncoderLength)}
	feedDict.update({decoderLabels[t]: decoderTargetTrain[t] for t in range(maxDecoderLength)})
	feedDict.update({decoderInputs[t]: decoderInputTrain[t] for t in range(maxDecoderLength)})
	feedDict.update({feedPrevious: False})

	curLoss, _, pred = sess.run([loss, optimizer, decoderPrediction], feed_dict=feedDict)
	
	if (i % 50 == 0):
		print('######################')
		print('Текущая потеря: ' + str(curLoss) + ' на итерации ' + str(i))
		print('######################')
		summary = sess.run(merged, feed_dict=feedDict)
		writer.add_summary(summary, i)
	if (i % 25 == 0 and i != 0):
		num = randint(0,len(encoderTestStrings) - 1)
		print ('- ' + translit(encoderTestStrings[num],'ru')) # Выводим пример вопроса с обратной транслитерацией
		inputVector = getTestInput(encoderTestStrings[num], wordList, maxEncoderLength);
		feedDict = {encoderInputs[t]: inputVector[t] for t in range(maxEncoderLength)}
		feedDict.update({decoderLabels[t]: zeroVector for t in range(maxDecoderLength)})
		feedDict.update({decoderInputs[t]: zeroVector for t in range(maxDecoderLength)})
		feedDict.update({feedPrevious: True})
		ids = (sess.run(decoderPrediction, feed_dict=feedDict))
		responseList = idsToSentence(ids, wordList)
		sys.stdout.write('- ')
		try:
			for word in responseList: sys.stdout.write(word.lower())
		except Exception:
			print(' ')
		print('')
		print('########')

	if (i % 10000 == 0 and i != 0):
		savePath = saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=i)
