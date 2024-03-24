from astropy.nddata import NDData, NDDataRef, StdDevUncertainty
import uncertainties as unc
#from uncertainties import unumpy as unp
import uncertainties.umath as umath
import numpy as np

def power_nddata(nddata_ref, power):
    # Ensure power is a float or int
    if not isinstance(power, (float, int)):
        raise TypeError("Power must be a float or int")

        # Create a new UncertainQuantity object with the data and uncertainty
    data_with_uncertainty = unc.ufloat(nddata_ref.data, nddata_ref.uncertainty.array)

    # Calculate the new data and uncertainty
    result_with_uncertainty = data_with_uncertainty**power

    # Extract the data and uncertainty from the result
    data_powered = result_with_uncertainty.nominal_value
    uncertainty_powered = result_with_uncertainty.std_dev
    unit_powered = nddata_ref.unit**power

    # Create a new NDData object with the powered data and uncertainty
    powered_nddata = NDData(
        data=data_powered, 
        uncertainty=StdDevUncertainty(uncertainty_powered), 
        unit=unit_powered)

    return NDDataRef(powered_nddata)
    #return NDDataRef(data=data_powered, uncertainty=StdDevUncertainty(uncertainty_powered))

def convert_nddata(nddata_ref, unit):
    #print('DATA, UNIT', nddata_ref.data, nddata_ref.unit)
    conversion_factor = (nddata_ref.unit/unit).to(1)
    new_data = nddata_ref.data * conversion_factor
    new_uncertainty = nddata_ref.uncertainty.array * conversion_factor

    return NDDataRef(data=new_data, uncertainty=StdDevUncertainty(new_uncertainty), unit=unit)


def log_nddata(nddata_ref):
    try:
        conv_data = convert_nddata(nddata_ref, 1)
    except:
        raise Exception('Can only take the log of something unitless')
    
    # Create a new UncertainQuantity object with the data and uncertainty
    data_with_uncertainty = unc.ufloat(conv_data.data, conv_data.uncertainty.array)
    #print(data_with_uncertainty, type(data_with_uncertainty))

    # Calculate the new data and uncertainty
    result_with_uncertainty = umath.log(data_with_uncertainty)
    #print('result_with_uncertainty:',result_with_uncertainty, type(result_with_uncertainty))
    #print(type(unc.ufloat(result_with_uncertainty)))

    # Extract the data and uncertainty from the result
    data_logged = result_with_uncertainty.nominal_value
    uncertainty_logged = result_with_uncertainty.std_dev

    ## Calculate the new unit
    #unit_logged = np.log(conv_data.unit)

    # Create a new NDData object with the logged data and uncertainty
    logged_nddata = NDData(
        data=data_logged,
        uncertainty=StdDevUncertainty(uncertainty_logged),
        #unit=unit_logged log has no units
    )

    return NDDataRef(logged_nddata)