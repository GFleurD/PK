type UnnormalisedInputVector = Tensor Real [6]
type InputVector = Tensor Real [6]

conc = 0
temp = 1
wbc = 2
age = 3
weight = 4
sex = 5

type OutputVector= Tensor Real [1]

meanScalingValues : UnnormalisedInputVector
meanScalingValues = [13.41985028, 37.73609288, 11.88956131, 50.64, 76.37743713,  0.6]

standardDeviationValues : UnnormalisedInputVector
standardDeviationValues =  [7.32717627, 0.625989, 2.54834216, 23.22477987, 14.33796805, 0.48989795]

normalise : UnnormalisedInputVector -> InputVector
normalise x = foreach i .
  (x ! i - meanScalingValues ! i) / (standardDeviationValues ! i)

@network
pk : InputVector -> OutputVector

normpk : UnnormalisedInputVector -> OutputVector
normpk x = pk (normalise x)

safeInput : InputVector -> Bool
safeInput x = 
    0 <= x ! conc <= 30 and
    36.5 <= x ! temp <= 40 and -- temps from dummy data based on a person being sick
    7.5 <= x ! wbc <= 20 and
    18 <= x ! age <= 89 and
    50 <= x ! weight <= 100 and
    -- 0 <= x ! sex <= 1
    ((x ! sex == 1 ) or (x ! sex == 0))

safeOutput : InputVector -> Bool
-- safeOutput x = let y = normpk x in 0 <= (x ! conc) + ((y ! 0)/30) <= 30
safeOutput x = -1 <= ((normpk x) ! 0)/30 + (x ! conc) <= 50


@property
safe: Bool
safe = forall x . safeInput x => safeOutput x

---------

unhealthyInput : InputVector -> Bool
unhealthyInput x = 
    10 <= x ! conc <= 30 and
    38 <= x ! temp <= 40 and -- temps from dummy data based on a person being sick
    12 <= x ! wbc <= 20 and
    18 <= x ! age <= 89 and
    50 <= x ! weight <= 100 and
    -- 0 <= x ! sex <= 1
    ((x ! sex == 1 ) or (x ! sex == 0))  

unhealthyOutput : InputVector -> Bool
-- safeOutput x = let y = normpk x in 0 <= (x ! conc) + ((y ! 0)/30) <= 30
unhealthyOutput x = 10 <= ((normpk x) ! 0)/30 + (x ! conc) <= 50

@property
unhealthy: Bool
unhealthy = forall x . unhealthyInput x => unhealthyOutput x

---------

healthyInput : InputVector -> Bool
healthyInput x = 
    0 <= x ! conc <= 30 and
    36 <= x ! temp <= 38 and -- temps from dummy data based on a person being sick
    4 <= x ! wbc <= 12 and
    18 <= x ! age <= 89 and
    50 <= x ! weight <= 100 and
    -- 0 <= x ! sex <= 1
    ((x ! sex == 1 ) or (x ! sex == 0))  

healthyOutput : InputVector -> Bool
-- safeOutput x = let y = normpk x in 0 <= (x ! conc) + ((y ! 0)/30) <= 30
healthyOutput x = (normpk x) ! 0  == 0

@property
healthy: Bool
healthy = forall x . healthyInput x => healthyOutput x

-------

healthydirection : InputVector -> Real
-- safeOutput x = let y = normpk x in 0 <= (x ! conc) + ((y ! 0)/30) <= 30
healthydirection x = let y = normpk x in (x ! temp) + 1 * (0.08 -0.005*((x ! conc) + ((y ! 0)/30)) - 0.12 * ((x ! temp) - 37 )) 

@property
healthydirectionhelp: Bool
healthydirectionhelp = forall x. safeInput x => healthydirection x <= (x ! temp)