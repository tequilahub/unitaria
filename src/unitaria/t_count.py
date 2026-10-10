# Adapted from arxiv:2605.31544
import bisect
from typing import Tuple

from unitaria.circuit import Circuit

import numpy as np
import tequila as tq

staircase_tanalpha = [
    1.000000000000000000,
    0.414213562373095090,
    0.350834874673672581,
    0.312876580442609076,
    0.212012989773690319,
    0.199053696635809713,
    0.137553374901583453,
    0.114743436552311937,
    0.110968924780318418,
    0.106108155871760396,
    0.095599761041589515,
    0.093857467246492854,
    0.089772198956584240,
    0.081007642852468559,
    0.080059368825502769,
    0.071823156445232433,
    0.068074811618681924,
    0.063557891353411389,
    0.056565932310567113,
    0.054687766545393701,
    0.043853474353372793,
    0.043184230632607387,
    0.041757558494795996,
    0.035525578867165862,
    0.026620221579097662,
    0.023740068332829375,
    0.023219765631210493,
    0.023041225970146049,
    0.021820124509931472,
    0.021584661590176659,
    0.021095252610628751,
    0.020170619047074331,
    0.016419959844448571,
    0.013250840260523361,
    0.013201874372092005,
    0.011678064337090286,
    0.011610273857837769,
    0.010123357001352220,
    0.009178937379670129,
    0.008939807749050285,
    0.008911641286055169,
    0.008891831473781506,
    0.008223547022012874,
    0.008134840314959944,
    0.005676564243201448,
    0.005490790602226593,
    0.005438659165343685,
    0.005011421293695884,
    0.004445849949711543,
    0.003802274290667123,
    0.003414481239748043,
    0.003400465772733619,
    0.003362918351896612,
    0.002623446891891916,
    0.002421456525235684,
    0.001942671784383428,
]

staircase_avg_t_count = [
    0.000000000000000000,
    1.414213562373094923,
    8.358523998839725522,
    14.90836057126281666,
    18.66666666666666430,
    35.41820150062702055,
    36.69143914061946532,
    45.81060988897981900,
    59.62423066692823426,
    79.37357444754127300,
    82.26631789232057201,
    91.53596103867951683,
    100.1969478876198707,
    103.3309855545793710,
    114.8865692342005786,
    129.9467535443398845,
    130.1425954431041134,
    137.7083130201563108,
    162.2185949989647042,
    177.3704252435253466,
    190.7728291979915696,
    234.8268288886290804,
    253.6524018919697312,
    274.1441976350207597,
    277.6535538936209377,
    337.6541634661525109,
    418.3216273097789895,
    546.1909215474321400,
    569.3392538941667453,
    610.1284182110173333,
    621.5720973965859457,
    648.8713606032779353,
    670.4052913138149279,
    869.0702760915087310,
    1061.190329568459902,
    1126.943755563780542,
    1206.586675638444831,
    1334.124470449861747,
    1395.777866999348362,
    1472.699658090156618,
    1580.987320181555106,
    1690.149558553591305,
    1831.306963333395743,
    1910.882476918264274,
    2018.310491944398791,
    2324.117154310660226,
    2486.215289670361472,
    3293.182410105669078,
    3605.545686445599586,
    4083.705682496006830,
    4431.716899082452983,
    4707.761035937885026,
    5205.974979638611330,
    6671.197660262269892,
    7123.787980357737979,
    8537.336639973320416,
]

staircase_phi = [
    0.785398163397448168,
    0.392699081698724195,
    0.255495373648521762,
    0.284924126622062401,
    0.192835807949161497,
    0.173552440720655482,
    0.124107795648728134,
    0.110082423147095237,
    0.109906358777430102,
    0.101528462773912573,
    0.091710893591349804,
    0.093403544968404958,
    0.070187845172285837,
    0.077752676679104127,
    0.078669855732249924,
    0.069493297706496718,
    0.057791644573391976,
    0.061888637249731267,
    0.055597998797776847,
    0.053665989739915342,
    0.041987261143899557,
    0.042637172618535335,
    0.041443167083348088,
    0.034682096952168145,
    0.019816715253174161,
    0.022220137471257971,
    0.022717871336363209,
    0.022893844150929561,
    0.021083637492522535,
    0.021313565565117158,
    0.020116119052836046,
    0.020040204171539192,
    0.016410933443297120,
    0.013234079689553997,
    0.013194264811155256,
    0.011536666869366122,
    0.011604021540317999,
    0.010119687083072504,
    0.008597823205676720,
    0.008827793348440431,
    0.008855693342139597,
    0.008875421608560755,
    0.008191240044674157,
    0.008111792864880870,
    0.005202477715701859,
    0.005378493332280404,
    0.005430047050166927,
    0.005010434945759594,
    0.004437666869261012,
    0.003795608808731670,
    0.003384718928535352,
    0.003398669385652424,
    0.003361547449303978,
    0.002623229168641762,
    0.002386380112804906,
    0.001932691904085085,
]

staircase_tanalpha = staircase_tanalpha[::-1]
staircase_avg_t_count = staircase_avg_t_count[::-1]
staircase_phi = staircase_phi[::-1]


def get_t_count_for_rot(theta: float, delta: float) -> Tuple[float, float]:
    """
    Parameters
    ----------
    theta : float
        The angle of the rotation.
    delta : float
        The desired value of delta (either the diamond norm error or lambda-1).

    Returns
    -------
    av_t_count : float
        The calculated average T gate count.
    delta_true : float
        The actual value of delta used. This might be less than the requested
        value of delta if the solution is from the staircase.
    """
    # Map theta to range [0, pi/8] by applying Clifford gates as needed.
    if np.abs(theta) > np.pi / 8:
        theta = (theta + np.pi / 8) % (np.pi / 4) - np.pi / 8
    theta = np.abs(theta)

    if theta == 0:
        return 0.0, 0.0

    tanalpha = delta / np.sin(2 * theta) + np.tan(theta)
    assert np.sin(2 * theta) != 0 and tanalpha != 0, "Error: accuracy too poor, sin(2 * theta) = 0 or tanalpha = 0"

    # Find index of largest tanalpha value that is smaller than or equal the given tanalpha.
    index = bisect.bisect_right(staircase_tanalpha, tanalpha) - 1
    if index >= 0 and np.tan(theta) <= staircase_phi[index]:
        # We have found a solution in the staircase that we can use.
        av_t_count = staircase_avg_t_count[index] * np.sin(2 * theta)
        delta_true = (staircase_tanalpha[index] - np.tan(theta)) * np.sin(2 * theta)
    else:
        # We did not find a solution in the staircase. Use the asymptotic formula instead.
        alpha = delta / (2 * theta) + theta
        K = (2 * np.sqrt(2 * np.e**3) / 3) ** (2 / 3)
        phi_0 = max(alpha - alpha / np.log(K / alpha), theta)
        av_t_count = (3 * theta / (alpha + 2 * phi_0)) * np.log2(12 / (((alpha - phi_0) ** 2) * (alpha + 2 * phi_0)))
        delta_true = delta

    # Finally, take the minimum of the above result and the angle-independent result.
    worst_case_t_count = 1.52 * np.log2(1 / delta) - 0.01
    if av_t_count > worst_case_t_count:
        av_t_count = worst_case_t_count
        delta_true = delta

    return av_t_count, delta_true


def get_t_count(circuit: Circuit, precision: float) -> int:
    # This is actually slightly cheating, since this way the error
    # of the circuit and sampling might add up to be larger than
    # precision, but since we only use it to count the gates, the
    # difference should only be logarithmic.
    compiler = tq.CircuitCompiler.all_flags_true()
    compiler.pauli_rotations = False
    compiler.toffoli = False
    compiled = compiler.compile_circuit(circuit._tq_circuit)

    t_count = 0

    for gate in compiled.gates:
        name = gate.name.lower()
        if name == "x" and len(gate.control) == 2:
            t_count += 7
        if name in ["rx", "ry", "rz"]:
            if abs(np.remainder(gate.parameter, np.pi / 2)) < precision:
                # Can be implemented without T
                continue
            elif abs(np.remainder(gate.parameter, np.pi / 4)) < precision:
                # This was probably intended to be a T gate
                t_count += 1
            else:
                # Factor 1/2 because of different angle convention
                t_count += get_t_count_for_rot(gate.parameter / 2, precision)[0]

    return t_count
