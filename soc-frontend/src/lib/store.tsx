import React, { createContext, useContext, useState, useCallback } from 'react';
import { Alert, PredictionLabel, AlertStatus, RemediationAction, RemediationResponse } from '../types/alert';
import { makeRemediationAction } from '../lib/remediation';

interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
}

interface AlertStore {
  alerts: Alert[];
  selectedAlert: Alert | null;
  toasts: Toast[];
  addAlert: (alert: Alert) => void;
  updateAlertStatus: (id: string, status: AlertStatus) => void;
  updateAlertPrediction: (id: string, prediction: PredictionLabel) => void;
  sendToRemediation: (id: string, remediationPrediction?: RemediationResponse, incidentFeatures?: number[]) => void;
  addRemediationAction: (alertId: string, type: RemediationAction['type']) => void;
  updateRemediationAction: (alertId: string, actionId: string, status: 'approved' | 'rejected', notes?: string) => void;
  setSelectedAlert: (alert: Alert | null) => void;
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  escalateAlert: (id: string) => void;
}

const AlertContext = createContext<AlertStore | null>(null);

export function AlertProvider({ children }: { children: React.ReactNode }) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = `toast-${Date.now()}`;
    setToasts(t => [...t, { ...toast, id }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000);
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts(t => t.filter(x => x.id !== id));
  }, []);

  const addAlert = useCallback((alert: Alert) => {
    setAlerts(prev => [alert, ...prev]);
  }, []);

  const updateAlertStatus = useCallback((id: string, status: AlertStatus) => {
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, status } : a));
    setSelectedAlert(prev => prev?.id === id ? { ...prev, status } : prev);
  }, []);

  const updateAlertPrediction = useCallback((id: string, prediction: PredictionLabel) => {
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, prediction, status: 'resolved' as AlertStatus } : a));
    setSelectedAlert(prev => prev?.id === id ? { ...prev, prediction, status: 'resolved' as AlertStatus } : prev);
  }, []);

  const sendToRemediation = useCallback((id: string, remediationPrediction?: RemediationResponse, incidentFeatures?: number[]) => {
    const actions: RemediationAction[] = [];
    if (remediationPrediction?.account_response.prediction) {
      actions.push(makeRemediationAction('disable_account'));
    }
    if (remediationPrediction?.endpoint_response.prediction) {
      actions.push(makeRemediationAction('isolate_endpoint'));
    }

    setAlerts(prev => prev.map(a => a.id === id ? {
      ...a,
      status: 'remediation' as AlertStatus,
      remediationPrediction,
      incidentFeatures: incidentFeatures || a.incidentFeatures,
      remediationActions: actions,
    } : a));
    setSelectedAlert(prev => prev?.id === id ? {
      ...prev,
      status: 'remediation' as AlertStatus,
      remediationPrediction,
      incidentFeatures: incidentFeatures || prev.incidentFeatures,
      remediationActions: actions,
    } : prev);
  }, []);

  const addRemediationAction = useCallback((alertId: string, type: RemediationAction['type']) => {
    const action = makeRemediationAction(type);
    setAlerts(prev => prev.map(a => a.id === alertId ? {
      ...a,
      remediationActions: [...(a.remediationActions || []), action]
    } : a));
  }, []);

  const updateRemediationAction = useCallback((alertId: string, actionId: string, status: 'approved' | 'rejected', notes?: string) => {
    setAlerts(prev => prev.map(a => a.id === alertId ? {
      ...a,
      remediationActions: a.remediationActions?.map(r =>
        r.id === actionId ? { ...r, status, notes } : r
      )
    } : a));
  }, []);

  const escalateAlert = useCallback((id: string) => {
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, status: 'escalated' as AlertStatus } : a));
    setSelectedAlert(prev => prev?.id === id ? { ...prev, status: 'escalated' as AlertStatus } : prev);
  }, []);

  return (
    <AlertContext.Provider value={{
      alerts, selectedAlert, toasts,
      addAlert, updateAlertStatus, updateAlertPrediction,
      sendToRemediation, addRemediationAction, updateRemediationAction,
      setSelectedAlert, addToast, removeToast, escalateAlert
    }}>
      {children}
    </AlertContext.Provider>
  );
}

export function useAlertStore() {
  const ctx = useContext(AlertContext);
  if (!ctx) throw new Error('useAlertStore must be used within AlertProvider');
  return ctx;
}
