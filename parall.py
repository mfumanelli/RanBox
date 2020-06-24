import os                                                                       
from multiprocessing import Pool                                                
import telegram                                                                
                                                                                
processes = ('perscript_adattivo.py', 'perscript_adattivo.py', 'perscript_adattivo.py', 'perscript_adattivo.py')                                    
                                                  
def notify_ending(message):
    token = '1116796554:AAEyr52UujgqbXkg1yfCvqn-oF4WMuMKgBw'
    chat_id = '299870514'
    bot = telegram.Bot(token=token)
    bot.sendMessage(chat_id=chat_id, text=message)

def run_process(process):                                                             
    os.system('python {}'.format(process))                                       
                                                                                
                                                                                
pool = Pool(processes=4)                                                        
pool.map(run_process, processes)

notify_ending('Il codice ha finito')