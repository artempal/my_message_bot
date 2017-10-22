# -*- coding: utf-8 -*-
import vk_api
import sys
import re  
import numpy as np
from transliterate import translit, get_available_language_codes


reload(sys)  
sys.setdefaultencoding('utf-8')

# Константы

DATA_FILE = 'data.txt'
DICT_FILE = 'conversationDictionary.npy'
CONSERV_FILE = 'conversationData.txt'
REG_FILTER = re.compile(u'[^а-яА-Я ]') # фильтр сообщений - по умолчанию только русские символы и пробелы

def vk_dataset():
    
    #login = raw_input('Введите логин: ')
    #password = raw_input('Введите пароль: ')
    login = ''
    password = ''
    my_id = 136771035

    vk_session = vk_api.VkApi(login, password)

    try:
        vk_session.auth()
    except vk_api.AuthError as error_msg:
        print(error_msg)
        return

    tools = vk_api.VkTools(vk_session)

    print 'Получаем диалоги...'
    dialogs = tools.get_all('messages.getDialogs', 200)

    # Получаем все id пользователей, с которыми переписывались
    all_user_id = []
    for dialog in dialogs['items']:
        if (dialog['message']['user_id']) > 0: # если это человек, а не сообщество
            all_user_id.append(dialog['message']['user_id']) # добавляем в массив

    data_file = open(DATA_FILE,'w')

    for user_id in all_user_id:
        data_file.write('Переписка с пользователем  ' + str(user_id) + '\n\n')
        messages = tools.get_all('messages.getHistory',200,{'user_id': user_id}) 
        for message in messages['items']:
            new_str = ''
            if(message['from_id'] == my_id):
                new_str += '# Я: '
            else:
                new_str += '# ' + str(user_id) + ': '
            message_body = clean_message(message['body'])
            if(len(message_body)==0): # если сообщение пустое
                continue
            new_str += message_body + '\n'
            data_file.write(new_str)
        data_file.write('\n\n')
    data_file.close()

def clean_message(message_body):
    message_body = message_body.replace('\n',' ').lower() # удаляем переносы строки
    message_body = REG_FILTER.sub(' ',message_body) # фильтруем сообщения
    message_body = message_body.strip() # удалим лишние пробелы в начале и конце сообщения
    message_body = re.sub(r'\s+', ' ', message_body) # удалим лишние пробелы внутри
    message_body = translit(message_body,'ru',reversed=True) #делаем транслитерацию
    return message_body



def transformation_data():
    print 'Начинаем обработку сообщений...'
    responseDictionary = dict()
    data_file = open(DATA_FILE, 'r') 
    allLines = data_file.readlines()
    myMessage, otherPersonsMessage, currentSpeaker = "","",""
    for index,lines in enumerate(allLines):
	    rightBracket = lines.find('#') + 2
	    justMessage = lines[rightBracket:]
	    colon = justMessage.find(':')
	    # Находим сообщение которое отправил я
	    if (justMessage[:colon] == 'Я'):
	        if not myMessage:
	            startMessageIndex = index - 1
	        myMessage += justMessage[colon+2:]
	        
	    elif myMessage:
	        # Смотрим сообщения человека
	        for counter in range(startMessageIndex, 0, -1):
	            currentLine = allLines[counter]
	            rightBracket = currentLine.find('#') + 2
	            justMessage = currentLine[rightBracket:]
	            colon = justMessage.find(':')
	            if not currentSpeaker:
	                currentSpeaker = justMessage[:colon]
	            elif (currentSpeaker != justMessage[:colon] and otherPersonsMessage):
	                # Понимаем, что начал говорить другой собеседник
	                otherPersonsMessage = otherPersonsMessage
	                myMessage = myMessage
	                responseDictionary[otherPersonsMessage] = myMessage
	                break
	            otherPersonsMessage = justMessage[colon+2:] + otherPersonsMessage
	        myMessage, otherPersonsMessage, currentSpeaker = "","",""    
    return responseDictionary 

def save_my_dict(trans_data):

    print 'Всего слов в словаре:', len(trans_data)

    print 'Сохраняем...'
    np.save(DICT_FILE, trans_data)

    conversationFile = open(CONSERV_FILE, 'w')
    for key,value in trans_data.iteritems():
        if (not key.strip() or not value.strip()):
            # если пустая строка
            continue
   	conversationFile.write(key.strip() + value.strip())
    conversationFile.close()


#vk_dataset()
trans_data = transformation_data()
save_my_dict(trans_data)
