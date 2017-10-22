# -*- coding: utf-8 -*-
import vk_api
import sys
import re  


reload(sys)  
sys.setdefaultencoding('utf-8')


def main():
    
    #login = raw_input('Введите логин: ')
    #password = raw_input('Введите пароль: ')
    login = ''
    password = ''
    my_id = 136771035
    reg_filter = re.compile(u'[^а-яА-Я ]') # фильтр сообщений - по умолчанию только русские символы и пробелы

    vk_session = vk_api.VkApi(login, password)

    try:
        vk_session.auth()
    except vk_api.AuthError as error_msg:
        print(error_msg)
        return

    tools = vk_api.VkTools(vk_session)

    dialogs = tools.get_all('messages.getDialogs', 200)

    # Получаем все id пользователей, с которыми переписывались
    all_user_id = []
    for dialog in dialogs['items']:
        if (dialog['message']['user_id']) > 0: # если это человек, а не сообщество
            all_user_id.append(dialog['message']['user_id']) # добавляем в массив

    data_file = open('data.txt','w')

    for user_id in all_user_id:
        data_file.write('Переписка с пользователем  ' + str(user_id) + '\n\n')
        messages = tools.get_all('messages.getHistory',200,{'user_id': user_id}) 
        for message in messages['items']:
            new_str = ''
            if(message['from_id'] == my_id):
                new_str += 'Я: '
            else:
                new_str += 'Собеседник: '
            message_body = reg_filter.sub(' ',message['body']) # фильтруем сообщения
            message_body = message_body.strip() # удалим лишние пробелы в начале и конце сообщения
            message_body = re.sub(r'\s+', ' ', message_body) # удалим лишние пробелы в центре
            if(len(message_body)==0): # если сообщение пустое
                continue
            new_str += message_body + '\n'
            data_file.write(new_str)
        data_file.write('\n\n')
    data_file.close()



main()