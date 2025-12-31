type UnnormalisedInputVector = Tensor Real [5]
type InputVector = Tensor Real [5]

conc = 0
temp = 1
wbc = 2
age = 3
weight = 4

type OutputVector= Tensor Real [1]

meanScalingValues : UnnormalisedInputVector
meanScalingValues = [2.01547446, 37.5597833, 10.14143758, 52.65, 76.00057268]

standardDeviationValues : UnnormalisedInputVector
standardDeviationValues =  [2.08296231, 0.77084898, 2.67494003, 21.48249287, 14.94338443]

normalise : UnnormalisedInputVector -> InputVector
normalise x = foreach i .
  (x ! i - meanScalingValues ! i) / (standardDeviationValues ! i)

@network
pk : InputVector -> OutputVector

normpk : UnnormalisedInputVector -> OutputVector
normpk x = pk (normalise x)

safeInput : InputVector -> Bool
safeInput x = 
    0 <= x ! conc <= 40 and
    36.5 <= x ! temp <= 40 and -- temps from dummy data based on a person being sick
    7.5 <= x ! wbc <= 20 and
    18 <= x ! age <= 90 and
    50 <= x ! weight <= 100

safeOutput : InputVector -> Bool
safeOutput x = 0 <= (((normpk x) ! 0)/(50*(50/70))) + (x ! conc) <= 40

@property
safe: Bool
safe = forall x . safeInput x => safeOutput x

unhealthyInput : InputVector -> Bool
unhealthyInput x = 
    0 <= x ! conc <= 40 and
    37 <= x ! temp <= 40 and -- temps from dummy data based on a person being sick
    8 <= x ! wbc <= 20 and
    18 <= x ! age <= 90 and
    50 <= x ! weight <= 100

unhealthyOutput : InputVector -> Bool
-- safeOutput x = let y = normpk x in 0 <= (x ! conc) + ((y ! 0)/30) <= 30
unhealthyOutput x = 0.0 <= ((normpk x) ! 0)

@property
unhealthy: Bool
unhealthy = forall x . unhealthyInput x => unhealthyOutput x

---------

healthyInput : InputVector -> Bool
healthyInput x = 
    0 <= x ! conc <= 40 and
    36.5 <= x ! temp <= 37.0 and -- temps from dummy data based on a person being sick
    4 <= x ! wbc <= 8 and
    18 <= x ! age <= 90 and
    50 <= x ! weight <= 100

healthyOutput : InputVector -> Bool
healthyOutput x = (normpk x) ! 0  <= 1

@property
healthy: Bool
healthy = forall x . healthyInput x => healthyOutput x

