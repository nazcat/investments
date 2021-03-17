import numpy as np
import matplotlib.pyplot as plt

monthly_compound = lambda x , r , t : x * (1 + r)**t
constant_monthly_compound = lambda x , r , t : np.sum( [ monthly_compound(x,r,j) for j in range(1,t+1) ] )

def compute_total_value_from_deposits( x0 , xmonthly , annual_percentage , n_months ):
    
        return monthly_compound( x0 , annual_percentage/100./12. , n_months ) + \
        constant_monthly_compound( xmonthly , annual_percentage/100./12. , n_months )
    
    
def main( x0 , xmonthly ):
    annual_rates = [ 0 , 4 , 8 , 12 , 16 , 20 ]
    months = np.arange( 1 , 48 )
    balance = []
    
    for (i,ri) in enumerate(annual_rates):
        
        balance.append( [ compute_total_value_from_deposits( x0 , xmonthly , ri , tj ) for tj in months ] )   
        if ri==0:
            plt.plot( months , balance[i] , 'r--' )
        else:
            plt.plot( months , balance[i] , 'b' )
    
    plt.grid()
    plt.legend( [ str(ri) + '%' for ri in annual_rates ] )


def example_buy_a_house( x0 , xmonthly , annual_rate_mean , annual_rate_std , xmortgage , nbuyhouse ):
    
    random_r = lambda : np.random.normal( annual_rate_mean , annual_rate_std )
    nmonths = 12*10
    months = np.arange( 1 , nmonths+1 )
    months_mortgage = np.arange( nbuyhouse , nmonths )
    value_deposits = np.array( [ compute_total_value_from_deposits( x0 , xmonthly , random_r() , tj ) for tj in months ] )
        
    net_value = np.zeros_like( months )
    net_value[0:months_mortgage[0]] = value_deposits[0:months_mortgage[0]]
    
    if ( xmonthly-xmortgage < 0 ):
        net_value[months_mortgage[0]:]  = [ monthly_compound( \
                                                value_deposits[months_mortgage[0]] , random_r()/100./12. , tj-months_mortgage[0] ) \
                                                for tj in months_mortgage ]
        net_value[months_mortgage[1]:] = net_value[months_mortgage[1:]] - np.array( [ constant_monthly_compound( \
                                                xmortgage-xmonthly , random_r()/100./12. , tj-months_mortgage[0] ) \
                                                for tj in months_mortgage[1:] ] )
    else:
        net_value[months_mortgage[0]:]  = [ compute_total_value_from_deposits( \
                                                value_deposits[months_mortgage[0]] , xmonthly-xmortgage , random_r() , tj-months_mortgage[0] ) for tj in months_mortgage ]
    return value_deposits , net_value



def mc_sample_houses( x0 , xmonthly , annual_rate_mean , annual_rate_std , xmortgage_mean , xmortgage_std , nbuyhouse_min , nbuyhouse_max , mc_samples ):
    
    mc_value = []
    
    for i in range( mc_samples ):
        
        xmort = np.random.normal( xmortgage_mean , xmortgage_std )
        _ , net_value_i = example_buy_a_house( x0 , xmonthly , annual_rate_mean , annual_rate_std , xmort , \
                                               np.random.choice( np.arange( nbuyhouse_min , nbuyhouse_max + 1 ) ) )
        mc_value.append( net_value_i )
    return np.array( mc_value )




def example_buy_a_house_2( worth_t0 , monthly_income_after_tax_t0 , monthly_expenses_not_rent , \
                           rent_before_house , investments_apr , income_apr , xmortgage , year_buy_house ):  
    
    nmonths = 12*10
    months = np.arange( 1 , nmonths+1 )
    months_mortgage = np.arange( year_buy_house , nmonths )
    xmonthly  = monthly_income_after_tax_t0 - ( monthly_expenses_not_rent + rent_before_house )
    value_deposits = np.array( [ compute_total_value_from_deposits( worth_t0 , xmonthly , investments_apr() , tj ) for tj in months ] ) 
    net_value = np.zeros_like( months )
    net_value[0:months_mortgage[0]] = value_deposits[0:months_mortgage[0]]
    
    if ( monthly_income_after_tax_t0 - ( monthly_expenses_not_rent + xmortgage ) < 0 ):
        
        net_value[months_mortgage[0]:]  = [ monthly_compound( \
                                                value_deposits[months_mortgage[0]] , investments_apr()/100./12. , tj-months_mortgage[0] ) \
                                                for tj in months_mortgage ]
        net_value[months_mortgage[1]:] = net_value[months_mortgage[1:]] - np.array( [ constant_monthly_compound( \
                                                ( monthly_expenses_not_rent + xmortgage ) - monthly_income_after_tax_t0 , \
                                                investments_apr()/100./12. , tj-months_mortgage[0] ) \
                                                for tj in months_mortgage[1:] ] )
        
    else:
        net_value[months_mortgage[0]:]  = [ compute_total_value_from_deposits( \
                                                value_deposits[months_mortgage[0]] , \
                                                monthly_income_after_tax_t0 - ( monthly_expenses_not_rent + xmortgage ) , \
                                                investments_apr() , tj-months_mortgage[0] ) for tj in months_mortgage ]
    
    return value_deposits , net_value


def mc_sample_houses_2( worth_t0 , monthly_income_after_tax_t0 , monthly_expenses_not_rent , \
                        rent_before_house , yearly_investments_apr , yearly_income_apr , monthly_mortgage , year_buy_house , mc_samples ):
    
    mc_value = []
    for i in range( mc_samples ):
        
        _ , net_value_i = example_buy_a_house_2( worth_t0 , monthly_income_after_tax_t0 , monthly_expenses_not_rent , \
                                                 rent_before_house , yearly_investments_apr , yearly_income_apr , monthly_mortgage() , year_buy_house() )
        mc_value.append( net_value_i )
        
    return np.array( mc_value )




def compute_total_house_price( monthly_mortgage_mean , apr_house , nyears ):
    
    total_deposits = monthly_mortgage_mean * np.sum( [ ( 1 + apr_house/12./100. )**j for j in range(0,nyears*12+1) ] )
    return total_deposits / (1 + apr_house/12./100.)**(nyears*12)


class Finances:
    
    def __init__(self):
        
        self.worth_t0                     = 84000
        self.monthly_income_after_tax_t0  = 7100
        self.monthly_expenses_not_rent    = 2900
        self.rent_before_house            = 2200
        self.yearly_investments_apr       = lambda : np.maximum( 0 , np.random.normal( 15 , 2.5 ) )
        self.yearly_income_apr            = lambda : np.maximum( 0 , np.random.normal( 5 , 2 ) )
        self.monthly_mortgage             = lambda : np.random.normal( 5200 , 100 )
        self.year_buy_house               = lambda : np.random.choice( np.arange( 24 , 49 ) )
        
    def produce_list_inputs_for_mc_sample_houses( self ):
        
        return [ self.worth_t0 , self.monthly_income_after_tax_t0 , self.monthly_expenses_not_rent , \
                 self.rent_before_house , self.yearly_investments_apr , self.yearly_income_apr , \
                 self.monthly_mortgage , self.year_buy_house ]

    def compute_mean_monthly_mortgage( self ):
        
        return np.mean( [ self.monthly_mortgage() for i in range(2048) ] )
        
    

if __name__ == '__main__':
    
    # x0       = 84000
    # xmonthly = 2000
    # monthly_rent = 2200
    # monthly_mortgage_mean = 4200 + 1000
    # monthly_mortgage_std  = 100
    
    # mc_values = mc_sample_houses( x0 , xmonthly , 15. , 2.5 , monthly_mortgage_mean-monthly_rent , monthly_mortgage_std , 24 , 48 , 2048 )
    
    finances  = Finances()
    
    mc_values = mc_sample_houses_2( *finances.produce_list_inputs_for_mc_sample_houses() , 2048 )
    
    mc_mean = np.mean( mc_values , axis=0 )
    mc_20  = np.percentile( mc_values , 20 , axis=0 )
    mc_80  = np.percentile( mc_values , 80 , axis=0 )
    mc_5   = np.percentile( mc_values , 5 , axis=0 )
    mc_95  = np.percentile( mc_values , 95 , axis=0 )
    
    plt.figure()
    
    xticklab = np.arange(1,len(mc_mean)+1)
    
    plt.plot( xticklab , mc_mean , 'k' )
    plt.fill_between( xticklab , mc_20 , mc_80 , color='b' , alpha=0.5 )
    plt.fill_between( xticklab , mc_5 , mc_95  , color='b' , alpha=0.5 )
    
    xticklab = np.arange(0,len(mc_mean)+1,12)
    xticklab[0] = 1
    
    plt.gca().set_xticks( xticklab )
    plt.gca().set_xticklabels( xticklab , fontsize=16)
    plt.gca().set_yticks( np.arange(0 , np.max(mc_95)+1 , 100000) )
    plt.gca().set_yticklabels( np.arange(0 , np.max(mc_95)+1 , 100000)/1000. , fontsize=16)
    
    plt.grid()

    apr_house = 3.5
    random_apr_house = lambda : np.maximum( 1. , np.random.normal( apr_house , 1.5 ) )
    nyrs_house = 30
    total_house_mean = compute_total_house_price( finances.compute_mean_monthly_mortgage() , apr_house , nyrs_house )
    
    plt.title( str('$') + '%.2f' %(total_house_mean/(1e6)) + str( 'M , ' ) + str(apr_house) + '% , ' + str(nyrs_house) + ' yrs' , fontsize=20 )
    plt.xlabel( 'Months' , fontsize=16 )
    plt.ylabel( 'Net Worth (thousands)' , fontsize=16 )
    
    plt.tight_layout()
    
    # total_house_mc = [ compute_total_house_price( monthly_mortgage_mean , random_apr_house() , nyrs_house ) for i in range(2048) ]
    
    # plt.figure()
    # plt.hist( total_house_mc )
    
    plt.show()
