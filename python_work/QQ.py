from qqbot import _bot as bot

def send_msg(receiver,msg):
    bot.Login(['ink','298731943'])
    bot.SendTo(receiver,msg)
    bot.stop()

if __name__ == '__main__':
    receiver='1393502463'
    msg='test(发自pycharm)'
    send_msg(receiver,msg)