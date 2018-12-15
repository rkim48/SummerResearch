import smtplib
import os
from email.mime.text import MIMEText

def email_results():

    credentials_dir = '/media/ExtHDD2/rk48Ext/data/TA023/data.txt'
    logfile_dir = '/home/rk48/Desktop/EpilepsyVIP/DCEpy/Features/BurnsStudy/Mutual_Information/log_file.txt'

    def read_credentials_from_file(file_name):
        username, password = [x.strip() for x in open(file_name).readlines()]
        return username, password

    username, password = read_credentials_from_file(credentials_dir)

    f = open(logfile_dir, 'r')
    msg = MIMEText(f.read())
    f.close()

    msg['Subject'] = 'Results Update!'
    msg['From'] = 'rkim1337@gmail.com'
    msg['To'] = 'rk48@rice.edu'

    s = smtplib.SMTP('smtp.gmail.com', port = 587)
    s.starttls()
    s.login(username, password)
    s.sendmail('rkim1337@gmail.com', ['rk48@rice.edu'], msg.as_string())
    s.quit()
    return
