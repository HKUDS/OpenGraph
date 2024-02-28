import smtplib
from email.mime.text import MIMEText
from email.header import Header

def SendMail(message, subject=None):
	return
	default_subject = 'Experimental Anomaly Report'
	if subject != None:
		default_subject += ':' + subject
	subject = default_subject

	sender = 'xxx@xx.com'
	receivers = ['xxx@xx.com']
	message = 'Dear Artificial Anomaly Investigator,\n\n' + 'I am writing to bring to your attention that an anomaly occurs and cannot be solved by your exception handler in the recent experiment. Please refer to the following message for details.\n\n' + message + '\n\nBest regards,\nIntelligent Experiment Assistant'
	
	message = MIMEText(message, 'plain', 'utf-8')
	message['Subject'] = Header(subject, 'utf-8')
	message['From'] = Header('intel_assistant<xxx@xxx.com>')
	message['To'] = Header('investigator<xxx@xxx.com>')

	mail_host = 'smtp.xxx.com'
	mail_user = 'xxx@xxx.com'
	mail_pass = 'xxx'

	smtpObj = smtplib.SMTP()
	smtpObj.connect(mail_host, 25)
	smtpObj.login(mail_user, mail_pass)
	smtpObj.sendmail(sender, receivers, message.as_string())