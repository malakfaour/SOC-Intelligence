import { Alert, RemediationAction } from '../types/alert';

export function makeRemediationAction(type: RemediationAction['type']): RemediationAction {
  const labels: Record<RemediationAction['type'], string> = {
    block_ip: 'Block Source IP',
    disable_account: 'Disable User Account',
    isolate_endpoint: 'Isolate Endpoint',
    custom: 'Custom Action',
  };
  return {
    id: `REM-${type}-${Date.now()}`,
    type,
    label: labels[type],
    status: 'pending',
    timestamp: new Date(),
  };
}

export function deriveIncidentFeatures(alert: Alert): number[] {
  const featureSlice = alert.features.slice(0, 12);
  const safeFeature = (index: number, fallback = 0) => featureSlice[index] ?? fallback;
  const alertCount = Math.max(1, Math.round(1 + alert.confidence * 4 + alert.prediction));
  const machineEntityCount = Math.min(alertCount, Math.max(1, Math.round((safeFeature(0) + safeFeature(1)) % (alertCount + 1))));
  const deviceContextCount = Math.min(alertCount, Math.max(machineEntityCount, Math.round((safeFeature(2) + safeFeature(3)) % (alertCount + 1))));
  const vmResourceCount = alert.protocol === 'RDP' || alert.port === 3389 ? 1 : Math.round(safeFeature(4) % 2);
  const dominantCategoryCode = alert.prediction === 2 ? 8 : alert.prediction === 1 ? 3 : 0;
  const maxSuspicionScore = Math.min(3, Math.max(0, Math.round(alert.confidence * 3)));
  const maxVerdictScore = alert.prediction === 2 ? 2 : alert.prediction === 1 ? 1 : 0;
  const uniqueEntityTypes = Math.max(1, Math.min(5, new Set(featureSlice.map(value => Math.round(value) % 5)).size));
  const hasProcessEntity = alert.port === 22 || alert.port === 3389 || safeFeature(5) > 50 ? 1 : 0;
  const hasFileEntity = safeFeature(6) > 40 ? 1 : 0;
  const hasMachineEntity = alert.protocol === 'TCP' || alert.protocol === 'SSH' || safeFeature(7) > 20 ? 1 : 0;

  return [
    alertCount,
    machineEntityCount,
    machineEntityCount / alertCount,
    deviceContextCount,
    deviceContextCount / alertCount,
    vmResourceCount,
    dominantCategoryCode,
    maxSuspicionScore,
    maxVerdictScore,
    Math.max(maxSuspicionScore, maxVerdictScore),
    uniqueEntityTypes,
    hasProcessEntity,
    hasFileEntity,
    hasMachineEntity,
  ];
}
