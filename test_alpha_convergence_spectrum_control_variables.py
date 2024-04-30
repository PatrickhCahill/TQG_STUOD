field_name = 'stream'
field_name = 'buoyancy'
field_name = 'pv'
rotation_flag = True 

flag_use_regularised_psi = False 
alphas = [16.0, 64.0, 128.0, 220.0]

rtqg_psi_name = '_reg' if (flag_use_regularised_psi and field_name == 'stream') else ''

def powerseries_outputname(alpha, field=field_name, rotation_flag=rotation_flag):
    return 'powerseries_alpha_{}_field_{}{}.npy'.format(alpha,field,rtqg_psi_name) if rotation_flag else 'powerseries_alpha_{}_field_{}{}_norotation.npy'.format(alpha,field,rtqg_psi_name)

def wavenumbers_outputname(alpha, field=field_name, rotation_flag=rotation_flag):
    return 'wavenumbers_for_powerseries_alpha_{}_field_{}{}.npy'.format(alpha,field, rtqg_psi_name) if rotation_flag else 'wavenumbers_for_powerseries_alpha_{}_field_{}{}_norotation.npy'.format(alpha,field,rtqg_psi_name)


#plt.savefig(workspace_none.output_name('test_spectrum_time_series.png'.format(wavenumber), "visuals"))
def spectrum_time_series_plot_outputname(field=field_name, rotation_flag=rotation_flag):
    return 'test_spectrum_time_series_field_{}{}.png'.format(field, rtqg_psi_name) if rotation_flag else 'test_spectrum_time_series_field_{}{}_norotation.png'.format(field, rtqg_psi_name)

#anim.save(workspace_none.output_name('test_spectrum_animation.gif', "visuals"), writer='imagemagick')
def spectrum_animation_outputname(field=field_name, rotation_flag=rotation_flag):
    return 'test_spectrum_animation_field_{}{}.gif'.format(field, rtqg_psi_name) if rotation_flag else 'test_spectrum_animation_field_{}{}_norotation.gif'.format(field, rtqg_psi_name)



def function_mean_value_outputname(field=field_name):
    return 'mean_value_field_{}.png'.format(field)
