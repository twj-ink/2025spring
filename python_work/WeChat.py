import itchat

itchat.auto_login(hotReload=True)
# itchat.send('hello',toUserName='@ae753a161c9d0807f5b967bc4548bcffe7207090e8917bf617a940365d3707eb')
# friends=itchat.get_friends()
#
# for friend in friends:
#     if friend['NickName'] in ('3,7-Dihydro.',):
#         # print(friend)
#         itchat.send('test33', toUserName=friend['UserName'])
#         print(f'Message to {friend['NickName']} is done.')
    # if friend['NickName']=='3,7-Dihydro.':
    #     itchat.send('祝新年快乐！巳巳如意！',toUserName=friend['UserName'])
    #     print(f'Message to {friend['NickName']} was sent.')
    #
groups=itchat.get_chatrooms()
# for group in groups:
    # print(group)
itchat.send('all', toUserName='@@930ac7244217cb7149d4dfe57d856774be07ea685693f134f8b272567907b69c')