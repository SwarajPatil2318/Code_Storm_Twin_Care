"""Replaceable provider contract; prototype transport is a backend inbox."""
from typing import Protocol


class NotificationProvider(Protocol):
    name:str
    def send_alert(self,alert:dict,patient:dict,notification_id:str)->dict: ...
    def get_delivery_status(self,delivery:dict)->str: ...
    def acknowledge(self,delivery:dict)->dict: ...


class TwinCareWearableAdapter:
    name='twincare_wearable_simulator'
    def send_alert(self,alert,patient,notification_id):
        if alert['patient_id']!=patient['patient_id']:raise ValueError('Recipient mismatch')
        return {'notification_id':notification_id,'patient_id':patient['patient_id'],'alert_id':alert['alert_id'],
            'status':'DELIVERED','transport':'backend_inbox','native_push':False}
    def get_delivery_status(self,delivery):return delivery['status']
    def acknowledge(self,delivery):return {'notification_id':delivery['notification_id'],'status':'ACKNOWLEDGED'}
