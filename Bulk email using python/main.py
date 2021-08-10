import smtplib
import mimetypes
import os
from app import app
from werkzeug.utils import secure_filename
from flask import flash, render_template, request

from email.message import EmailMessage
from email.utils import make_msgid
from email.mime.base import MIMEBase 
from email import encoders 
		
@app.route('/')
def email_page():
	return render_template('emails.html')
	
@app.route('/send', methods=['POST'])
def send_email():
    sender_email = request.form['senders-id']
    sender_password = request.form['senders-app-password']
    subject = request.form['email-subject']
    body = request.form['email-body']
    emails = request.form['emails']
    emails_l = emails.split(',')
    file = request.form['email-attachment']
    filename= request.form['email-attachment1']
    e_length= len(emails_l)
    print(e_length)
    if subject and body and emails:

                if (e_length > 60):
                        flash('Email Not send because email list is greater than 60')
                        return render_template('emails.html', color='green')
                else:
                        for i in emails_l:
                                msg = EmailMessage()
                                print("Sendig mail to " +i)

                                asparagus_cid = make_msgid()
                                    
                                msg['Subject'] = subject
                                msg['From'] = sender_email
                                msg['To'] = i
                                    
                                msg.add_alternative(body.format(asparagus_cid=asparagus_cid[1:-1]), subtype='html')

                                if (file !=''):
                                        with open(file, "rb") as attachment:
                                                part = MIMEBase("application", "octet-stream")
                                                part.set_payload(attachment.read())
                                        encoders.encode_base64(part)
                                        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
                                        msg.attach(part)
                                                    
                                s = smtplib.SMTP('smtp.mail.yahoo.com', 587)

                                s.starttls()

                                s.login(sender_email, sender_password)
                                s.send_message(msg)
                                s.quit()
                                flash('Email to '+ i+ ' successfully sent to recepients')
                         
            
        
                        return render_template('emails.html', color='green')
    else:
        flash('Email subject, body and list of emails are required field')
        return render_template('emails.html', color='red')

if __name__ == "__main__":
    app.run()
