from typing import List, Dict, Callable, Optional, Union
from datetime import datetime
from enum import Enum

from utils.logger import setup_logger

logger = setup_logger("alerts")

class AlertCondition(str, Enum):
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUAL = "eq"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"

class Alert:
    """Individual alert configuration"""
    
    def __init__(
        self,
        name: str,
        metric: str,
        condition: AlertCondition,
        threshold: float,
        callback: Callable[['Alert', float], None],
        cooldown_seconds: int = 60
    ):
        self.name = name
        self.metric = metric  # e.g., 'zscore', 'correlation', 'price'
        self.condition = condition
        self.threshold = threshold
        self.callback = callback
        self.cooldown_seconds = cooldown_seconds
        self.is_active = True
        self.triggered_at: Optional[datetime] = None
        self.last_value: Optional[float] = None
        self.trigger_count = 0
    
    def check(self, value: float) -> bool:
        """
        Check if alert condition is met
        
        Returns:
            True if alert triggered
        """
        if not self.is_active:
            return False
        
        # Cooldown check
        if self.triggered_at:
            elapsed = (datetime.now() - self.triggered_at).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False
        
        triggered = False
        
        if self.condition == AlertCondition.GREATER_THAN:
            triggered = value > self.threshold
            
        elif self.condition == AlertCondition.LESS_THAN:
            triggered = value < self.threshold
            
        elif self.condition == AlertCondition.EQUAL:
            triggered = abs(value - self.threshold) < 0.01
            
        elif self.condition == AlertCondition.CROSSES_ABOVE:
            if self.last_value is not None:
                triggered = self.last_value <= self.threshold < value
                
        elif self.condition == AlertCondition.CROSSES_BELOW:
            if self.last_value is not None:
                triggered = self.last_value >= self.threshold > value
        
        self.last_value = value
        
        if triggered:
            self.triggered_at = datetime.now()
            self.trigger_count += 1
            logger.info(f"Alert triggered: {self.name} (value={value:.4f}, threshold={self.threshold})")
            self.callback(self, value)
        
        return triggered
    
    def to_dict(self) -> Dict:
        """Serialize alert to dictionary"""
        return {
            'name': self.name,
            'metric': self.metric,
            'condition': self.condition.value,
            'threshold': self.threshold,
            'is_active': self.is_active,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'trigger_count': self.trigger_count
        }


class AlertManager:
    """
    Manages multiple alerts and alert history
    """
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_history: List[Dict] = []
        logger.info("AlertManager initialized")
    
    def add_alert(
        self,
        name: str,
        metric: str,
        condition: Union[str, AlertCondition],
        threshold: float,
        callback: Optional[Callable[['Alert', float], None]] = None
    ) -> str:
        """
        Add new alert
        
        Args:
            name: Alert name
            metric: Metric to monitor ('zscore', 'correlation', etc.)
            condition: Alert condition ('gt', 'lt', 'eq', etc.)
            threshold: Threshold value
            callback: Function to call when alert triggers
        
        Returns:
            Alert ID
        """
        if callback is None:
            callback = self._default_alert_callback

        # Accept either an AlertCondition or a string; validate input
        if isinstance(condition, str):
            try:
                condition_enum = AlertCondition(condition)
            except ValueError:
                raise ValueError(f"Invalid alert condition: {condition}")
        else:
            condition_enum = condition

        alert = Alert(
            name=name,
            metric=metric,
            condition=condition_enum,
            threshold=threshold,
            callback=callback
        )
        
        self.alerts.append(alert)
        alert_id = f"alert_{len(self.alerts)}"
        
        logger.info(f"Alert added: {name} ({condition} {threshold})")
        return alert_id
    
    def check_alerts(self, metrics: Dict[str, float]):
        """
        Check all alerts against current metrics
        
        Args:
            metrics: Dict of metric_name -> value
        """
        for alert in self.alerts:
            if alert.metric in metrics:
                alert.check(metrics[alert.metric])
    
    def _default_alert_callback(self, alert: Alert, value: float) -> None:
        """Default callback that logs to history"""
        self.alert_history.append({
            'alert_name': alert.name,
            'metric': alert.metric,
            'value': value,
            'threshold': alert.threshold,
            'timestamp': datetime.now().isoformat()
        })

    def get_alert(self, alert_name: str) -> Optional[Alert]:
        """Return an Alert by name, or None if not found"""
        for a in self.alerts:
            if a.name == alert_name:
                return a
        return None
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history[-limit:]
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.alerts if alert.is_active]
    
    def remove_alert(self, alert_name: str) -> bool:
        """Remove alert by name"""
        alert = self.get_alert(alert_name)
        if alert is None:
            return False
        self.alerts.remove(alert)
        logger.info(f"Alert removed: {alert_name}")
        return True
    
    def toggle_alert(self, alert_name: str) -> bool:
        """Toggle alert active status"""
        alert = self.get_alert(alert_name)
        if alert is None:
            return False
        alert.is_active = not alert.is_active
        logger.info(f"Alert {'activated' if alert.is_active else 'deactivated'}: {alert_name}")
        return True
    
    def clear_history(self):
        """Clear alert history"""
        self.alert_history.clear()
        logger.info("Alert history cleared")
    
    def clear_all_alerts(self):
        """Remove all alerts"""
        count = len(self.alerts)
        self.alerts.clear()
        logger.info(f"Cleared {count} alerts")
```
