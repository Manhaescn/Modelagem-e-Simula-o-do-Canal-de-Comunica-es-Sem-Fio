
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D



#--------------------------------  parte das funcoes -------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_espalhamento_mult(ambiente, condicao, freq_portadora, seed=None ):
	if seed is not None:
		np.random.seed(seed)  # fixa a aleatoriedade

	# calculando para o ambiente UMi
	if ambiente.lower() == 'umi':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':
			mu = -0.24 * np.log10( 1 + freq_portadora ) -7.14
			sigma = 0.38

			# retorna o valor da amostra aleatoria linearizada
			return 10**( np.random.normal( mu, sigma ) )				

		# calculando para a condicao de NLos
		mu = -0.24 * np.log10( 1 + freq_portadora ) -6.83
		sigma = -0.16 * np.log10( 1 + freq_portadora ) + 0.28

		# retorna o valor da amostra aleatoria linearizada
		return 10**( np.random.normal( mu, sigma ) )


	# Calculando para o ambiente UMa
	if ambiente.lower() == 'uma':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':
			mu = -0.0963 * np.log10( 1 + freq_portadora ) -6.955
			sigma = 0.66

			# retorna o valor da amostra aleatoria linearizada
			return 10**( np.random.normal( mu, sigma ) )

		# calculando para a condicao de NLos
		mu = -0.204 * np.log10( 1 + freq_portadora ) -6.28
		sigma = 0.39

		# retorna o valor da amostra aleatoria linearizada
		return 10**( np.random.normal( mu, sigma ) )



def get_fator_prop( ambiente, condicao ):

	# calculando para o ambiente UMi
	if ambiente.lower() == 'umi':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':

			# achar o fator de proporcionalidade
			return 3


		# calculando para a condicao de NLos

		# achar o fator de proporcionalidade
		return 2.1


	# Calculando para o ambiente UMa
	if ambiente.lower() == 'uma':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':

			# achar o fator de proporcionalidade
			return 2.5


		# calculando para a condicao de NLos

		# achar o fator de proporcionalidade
		return 2.3


def get_atraso_mult( fator_prop, sigma, M, seed=None ):
	if seed is not None:
		np.random.seed(seed)  # fixa a aleatoriedade

	# achar a media 
	mu = fator_prop * sigma

	# gerar atrasos exponenciais
	tau_ii = np.random.exponential( mu, M )

	# normaliza e ordena os atrasos exponenciais
	return np.sort( tau_ii - min(tau_ii) )



def get_sombreamento( ambiente, condicao, M, seed=None ):
	if seed is not None:
		np.random.seed(seed)  # fixa a aleatoriedade

	# calculando para o ambiente UMi
	if ambiente.lower() == 'umi':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':

			# calcular o espalhamento
			return np.random.normal( 0, 4, M )

		# calculando para a condicao de NLos

		# calcular o espalhamento
		return np.random.normal( 0, 7.82, M )


	# Calculando para o ambiente UMa
	if ambiente.lower() == 'uma':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':

			# calcular o espalhamento
			return np.random.normal( 0, 4, M )

		# calculando para a condicao de NLos

		# calcular o espalhamento
		return np.random.normal( 0, 6, M )



def get_fator_rice( ambiente, condicao, seed=None ):
	if seed is not None:
		np.random.seed(seed)  # fixa a aleatoriedade

	# calculando para o ambiente UMi
	if ambiente.lower() == 'umi':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':

			# calcula o fator de rice ja linearizado
			return 10**( np.random.normal( 9, 5 ) / 10 )

		# calculando para a condicao de NLos
		return 0


	# Calculando para o ambiente UMa
	if ambiente.lower() == 'uma':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':	

			# calcula o fator de rice ja linearizado
			return 10**( np.random.normal( 9, 3.5 ) / 10 )


		# calculando para a condicao de NLos
		return 0



def get_potencia( condicao, sigma, fator_prop, tau, sombreamento, fator_rice ):

	alfa_i = np.exp( -tau * ( ( fator_prop - 1 ) / ( fator_prop * sigma ) ) ) * 10**( -sombreamento / 10 )

	alfa = ( 1 / ( fator_rice + 1 ) ) * ( alfa_i / np.sum( alfa_i[ 1: ] ) )
	
	if condicao.lower() == 'los':

		alfa[ 0 ] = fator_rice / ( fator_rice + 1 )
		
		return alfa

	else: 
 		return alfa



def get_espalhamento_mult_def( potencia, tau ):

	tau_med =  np.sum( tau * potencia / np.sum( potencia ))  
	return np.sqrt( np.sum( potencia * ( tau - tau_med )**( 2 ) ) / np.sum( potencia ) )



def get_espalhamento_ang_azimutal( ambiente, condicao, freq_portadora, seed=None ):
	if seed is not None:
		np.random.seed(seed)  # fixa a aleatoriedade

	# calculando para o ambiente UMi
	if ambiente.lower() == 'umi':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':
			mu = -0.08 * np.log10( 1 + freq_portadora ) + 1.73
			sigma = 0.014 * np.log10( 1 + freq_portadora ) + 0.28

			# retorna o valor da amostra espalhamento angular azimutal linearizada e em radianos
			return np.deg2rad( 10**( np.random.normal( mu, sigma ) ) ) 				

		# calculando para a condicao de NLos
		mu = -0.08 * np.log10( 1 + freq_portadora ) + 1.81
		sigma = 0.05 * np.log10( 1 + freq_portadora ) + 0.3

		# retorna o valor da amostra espalhamento angular azimutal linearizada e em radianos
		return np.deg2rad( 10**( np.random.normal( mu, sigma ) ) ) 


	# Calculando para o ambiente UMa
	if ambiente.lower() == 'uma':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':
			mu = 1.81
			sigma = 0.2

			# retorna o valor da amostra espalhamento angular azimutal linearizada e em radianos
			return np.deg2rad( 10**( np.random.normal( mu, sigma ) ) )

		# calculando para a condicao de NLos
		mu = -0.27 * np.log10( freq_portadora ) + 2.08
		sigma = 0.11

		# retorna o valor da amostra espalhamento angular azimutal linearizada e em radianos
		return np.deg2rad( 10**( np.random.normal( mu, sigma ) ) )



def get_angulo_azimutal( espalhamento_ang_azimutal, potencia, M, seed=None ):
	if seed is not None:
		np.random.seed(seed)  # fixa a aleatoriedade

	# calculo dos angulos finais
	theta_prime = 1.42 * espalhamento_ang_azimutal * np.sqrt(-np.log( potencia / np.max( potencia ) ) )
	U_theta = np.random.choice([-1, 1], size=M)
	Y_theta = np.random.normal(0, espalhamento_ang_azimutal / 7, size=M)

	theta = U_theta * theta_prime + Y_theta

	# caso existir visada direta (LoS)
	if condicao.lower() == 'los':

		return theta - theta[0]

	# caso seja NLoS
	return theta



def get_espalhamento_ang_elevacao( ambiente, condicao, freq_portadora, seed=None ):
	if seed is not None:
		np.random.seed(seed)  # fixa a aleatoriedade

	# calculando para o ambiente UMi
	if ambiente.lower() == 'umi':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':
			mu = -0.1 * np.log10( 1 + freq_portadora ) + 0.73
			sigma = -0.04 * np.log10( 1 + freq_portadora ) + 0.34

			# retorna o valor da amostra espalhamento angular de elevacao linearizada e em radianos
			return np.deg2rad( 10**( np.random.normal( mu, sigma ) ) ) 				

		# calculando para a condicao de NLos
		mu = -0.04 * np.log10( 1 + freq_portadora ) + 0.92
		sigma = -0.07 * np.log10( 1 + freq_portadora ) + 0.41

		# retorna o valor da amostra espalhamento angular de elevacao linearizada e em radianos
		return np.deg2rad( 10**( np.random.normal( mu, sigma ) ) )


	# Calculando para o ambiente UMa
	if ambiente.lower() == 'uma':

		# calculando para a condicao de Los
		if condicao.lower() == 'los':
			mu = 0.95
			sigma = 0.16

			# retorna o valor da amostra espalhamento angular de elevacao linearizada e em radianos
			return np.deg2rad( 10**( np.random.normal( mu, sigma ) ) )

		# calculando para a condicao de NLos
		mu = -0.3236 * np.log10( freq_portadora ) + 1.512
		sigma = 0.16

		# retorna o valor da amostra espalhamento angular de elevacao linearizada e em radianos
		return np.deg2rad( 10**( np.random.normal( mu, sigma ) ) )



def get_angulo_elevacao( espalhamento_ang_elevacao, potencia, M, seed=None ):
	if seed is not None:
		np.random.seed(seed)  # fixa a aleatoriedade

	# calculo dos angulos finais
	phi_i = -espalhamento_ang_elevacao * np.log(potencia / np.max( potencia ) )
	U_phi = np.random.choice( [-1, 1], size=M )
	Y_phi = np.random.normal( 0, espalhamento_ang_elevacao / 7, size=M )

	phi_barra = np.pi / 4  # valor médio arbitrário
	phi = U_phi * phi_i + Y_phi
	# caso existir visada direta (LoS)
	if condicao.lower() == 'los':

		return phi -phi[0] + phi_barra

	# caso seja NLoS
	return phi



def get_vetor_direcao_chegada( angulo_azimutal, angulo_elevacao ):

	return np.array( [np.cos( angulo_azimutal ) * np.sin( angulo_elevacao ), np.sin( angulo_azimutal ) * np.sin( angulo_elevacao ), np.cos( angulo_elevacao ) ] )


def get_desvio_doppler( vrx, freq_portadora, vetor_direcao_chegada, angulo_azimutal_v, angulo_elevacao_v ):

	Vrx = vrx * np.array([ np.cos(angulo_azimutal_v) * np.sin(angulo_elevacao_v), np.sin(angulo_azimutal_v) * np.sin(angulo_elevacao_v), np.cos(angulo_elevacao_v) ])
	desvio_doppler = 1 / ( 299792458/(freq_portadora* 10**9) ) * np.dot( vetor_direcao_chegada , Vrx )	
	return desvio_doppler



# ----------------------------------- area de calculo -----------------------------------------------------------------------------------------------------------------------------------------------------

seed = None

ambiente = 'UMi' 
condicao = 'Los'
N  = 100
fc = 3 # em GHz

sigma = get_espalhamento_mult( ambiente, condicao, fc, seed )

fator_prop = get_fator_prop( ambiente, condicao )

tau = get_atraso_mult( fator_prop, sigma, N, seed )

sombreamento = get_sombreamento( ambiente, condicao, N, seed )

fator_rice = get_fator_rice( ambiente, condicao, seed )

potencia = get_potencia( condicao, sigma, fator_prop, tau, sombreamento, fator_rice )

sigma_def = get_espalhamento_mult_def( potencia, tau )

espalhamento_ang_azimutal = get_espalhamento_ang_azimutal( ambiente, condicao, fc, seed )

theta = get_angulo_azimutal( espalhamento_ang_azimutal, potencia, N, seed )

espalhamento_ang_elevacao = get_espalhamento_ang_elevacao( ambiente, condicao, fc, seed )

phi = get_angulo_elevacao( espalhamento_ang_elevacao, potencia, N, seed )

direcao_chegada = get_vetor_direcao_chegada( theta, phi )



# -------------------------- parte das plotagem dos graficos ------------------------------------------------------------------------------------------------------------------------------------



# curvas da média e do desvio padrão do espalhamento de atraso στ como funções da frequência

# media
def plot_media_espalhamento_mult(fc):
	fc = np.logspace(0, 2, 100)  # de 10^0 (1 GHz) a 10^2 (100 GHz)

	UMi_LoS  = -0.24 * np.log10(1 + fc ) - 7.14
	UMi_NLoS = -0.24 * np.log10(1 + fc ) - 6.83
	UMa_LoS  = -0.0963 * np.log10(1 + fc ) - 6.955
	UMa_NLoS = -0.204 * np.log10(1 + fc ) - 6.28

	# Plot
	plt.figure(figsize=(8, 4))

	plt.plot(fc, 10**(UMi_LoS) * 10**6, 'b-', label='UMi-Los')
	plt.plot(fc, 10**(UMi_NLoS) * 10**6, 'r-', label='UMi-NLos')
	plt.plot(fc, 10**(UMa_LoS) * 10**6, 'b--', label='UMa-Los')
	plt.plot(fc, 10**(UMa_NLoS) * 10**6, 'r--', label='UMa-NLos')

	# Escalas logarítmicas
	plt.xscale('log')

	# Eixos e legendas
	plt.xlabel('Frequência da portadora – $f_c$ (GHz)', fontsize=12)
	plt.ylabel('Média de $\\sigma_\\tau$ ($\\mu$s)', fontsize=12)
	plt.legend()
	plt.grid(True, which="both", ls=":", linewidth=0.5)

	plt.tight_layout()
	plt.show()

# desvio
def plot_desvio_espalhamento_mult(fc):
	fc = np.logspace(0, 2, 100)  # de 10^0 (1 GHz) a 10^2 (100 GHz)

	UMi_LoS  = 0.38
	UMi_NLoS = -0.16 * np.log10(1 + fc ) + 0.28
	UMa_LoS  = 0.66
	UMa_NLoS = 0.39

	# Plot
	plt.figure(figsize=(8, 4))

	plt.plot( fc, np.full_like( fc, 10**(UMi_LoS) * 10**6 ) , 'b-', label='UMi-Los')
	plt.plot( fc, 10**(UMi_NLoS) * 10**6 , 'r-', label='UMi-NLos')
	plt.plot( fc, np.full_like( fc, 10**(UMa_LoS) * 10**6 ) , 'b--', label='UMa-Los')
	plt.plot(fc, np.full_like( fc, 10**(UMa_NLoS) * 10**6 ) , 'r--', label='UMa-NLos')

	# Escalas logarítmicas
	plt.xscale('log')

	# Eixos e legendas
	plt.xlabel('Frequência da portadora – $f_c$ (GHz)', fontsize=12)
	plt.ylabel('Desvio de $\\sigma_\\tau$ ($\\mu$s)', fontsize=12)
	plt.legend()
	plt.grid(True, which="both", ls=":", linewidth=0.5)

	plt.tight_layout()
	plt.show()



# perfil de atraso de potência do canal.

def plot_atraso_de_potencia(tau, potencia):
	# Criando o gráfico de setas
	(markerline, stemlines, baseline) = plt.stem(tau * 10**6, potencia, linefmt='k-', markerfmt='k^', basefmt=" ")

	# Ajustando log no eixo y
	plt.yscale("log")

	# Seta azul
	plt.stem(tau[0], potencia[0], linefmt='b-', markerfmt='b^', basefmt=" ")

	# Rótulos
	plt.xlabel('Domínio de Atraso – $\\tau$ ($\\mu$s)', fontsize=12)
	plt.ylabel('PDP', fontsize=12)

	# Grades
	plt.grid(True, which="both", linestyle=':', linewidth=0.5)

	plt.tight_layout()
	plt.show()



# gráficos da dispersão da potência nos domı́nios dos ângulos de chegada

# azimutal
def plot_potencia_angular_azimutal(theta, potencia):

	# Criando figura polar
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='polar')

	# Um ponto azul
	theta_azul = theta[0]
	r_azul = 10* np.log10( potencia[0] )
	ax.plot(theta_azul, r_azul, 'bo', markerfacecolor='none', markersize=8)  # ponto azul contornado

	# Pontos vermelhos
	theta_vermelho = theta[ 1: ]
	r_vermelho = 10 * np.log10( potencia[ 1: ] )
	ax.plot(theta_vermelho, r_vermelho, 'ro', markerfacecolor='none', markersize=5)

	# Exibe
	ax.set_rlim(r_vermelho.min(), 1)
	plt.show()


	# Gráfico de setas pretas
	plt.stem( np.rad2deg( theta[ 1: ] ), potencia[ 1: ], linefmt='k-', markerfmt='k^', basefmt=" ")

	# Adiciona o caminho direto (azimute = 0, potência = 1e-4 por ex)
	plt.stem( np.rad2deg( theta[0] ), potencia[0], linefmt='b-', markerfmt='b^', basefmt=" ")

	# Escala logarítmica no eixo y
	plt.yscale("log")

	# Rótulos
	plt.xlabel('Ângulos de chegada em azimute (°)', fontsize=12)
	plt.ylabel('Potência', fontsize=12)

	plt.grid(True, which="both", linestyle=':', linewidth=0.5)
	plt.tight_layout()
	plt.show()


# elevacao
def plot_potencia_angular_elevacao(phi, potencia):
	# Criando figura polar
	fig = plt.figure()
	ax = fig.add_subplot(111, polar=True)

	# Um ponto azul
	phi_azul = phi[0]
	r_azul = 10 * np.log10( potencia[0] )
	ax.plot(phi_azul, r_azul, 'bo', markerfacecolor='none', markersize=8)  # ponto azul contornado

	# Pontos vermelhos 
	phi_vermelho = phi[ 1: ]
	r_vermelho = 10 * np.log10( potencia[ 1: ] )
	ax.plot(phi_vermelho, r_vermelho, 'ro', markerfacecolor='none', markersize=5)

	# Exibe
	ax.set_rlim(r_vermelho.min(), 1) 
	plt.show()


	# Gráfico de setas pretas
	plt.stem( np.rad2deg( phi[ 1: ] ), potencia[ 1: ], linefmt='k-', markerfmt='k^', basefmt=" ")

	# Adiciona o caminho direto (elevacao = 0, potência = 1e-4 por ex)
	plt.stem( np.rad2deg( phi[0] ), potencia[0], linefmt='b-', markerfmt='b^', basefmt=" ")

	# Escala logarítmica no eixo y
	plt.yscale("log")

	# Rótulos
	plt.xlabel('Ângulos de chegada em elevacao (°)', fontsize=12)
	plt.ylabel('Potência', fontsize=12)

	plt.grid(True, which="both", linestyle=':', linewidth=0.5)
	plt.tight_layout()
	plt.show()



# vetores de direção de chegada 

def plot_direcao_chegada(direcao_chegada):
	# Cria uma figura 3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')


	for i in range(100):
		ax.plot([ 0, direcao_chegada[0][i] ], [ 0, direcao_chegada[1][i] ]  , [ 0, direcao_chegada[2][i] ], 'r-', marker='^', markerfacecolor='none')

	# --------- Caminho azul ---------
	ax.plot([ 0, direcao_chegada[0][0] ], [ 0, direcao_chegada[1][0] ], [ 0, direcao_chegada[2][0] ], 'b-', marker='^', markerfacecolor='none')

	# Rótulos
	ax.set_xlabel("Eixo X")
	ax.set_ylabel("Eixo Y")
	ax.set_zlabel("Eixo Z")

	ax.set_box_aspect([1,1,1])  # Proporção igual
	plt.tight_layout()
	plt.show()



# gráficos da dispersão da potência no domı́nio de desvio 

def plot_desvio_doppler( vrx, desvio_doppler, potencia ):
	plt.stem(desvio_doppler[0], potencia[0], linefmt='b-', markerfmt='k^', basefmt=" ")
	#for i in range(numero_coordenada):
	plt.stem(desvio_doppler[ 1: ], potencia[ 1: ], linefmt='k-', markerfmt='k^', basefmt=" ")

	# Escala logarítmica no eixo y
	plt.yscale('log')

	# Rótulos
	plt.xlabel("Desvio Doppler – $\\nu$ (Hz)", fontsize=12)
	plt.ylabel("Espectro Doppler", fontsize=12)

	# Texto no topo
	plt.text(-4, 3, f"vrx = {vrx} m/s, fc = {fc} GHz", fontsize=12)

	plt.grid(True, which="both", linestyle=":", linewidth=0.5)
	plt.show()



# gerar o sinal transmitido e o sinal recebido

def gerar_pulso(t, delta_t):
    return np.where((t >= 0) & (t <= delta_t), 1.0, 0.0)

def gerar_canal_multipercurso(N, delta_t, tau, fc, desvio_doppler, potencia):

    t = np.linspace(0, 5 * delta_t, 1000)
    r_t = np.zeros_like(t, dtype=complex)
  
    for n in range(N):
        s_t_n = gerar_pulso(t - tau[n], delta_t)
        r_t += ( potencia[n] )**(0.5) * np.exp( -1j * 2*np.pi * ( (fc * 10**9 + desvio_doppler[n]) * tau[n] - desvio_doppler[n] * t ) ) * s_t_n
    return t, r_t

def plot_sinal_recebido():
	delta_ts = [1 * 10**(-7), 1 * 10**(-5), 1 * 10**(-3)]
	plt.figure(figsize=(10, 6))
	for i, dt in enumerate(delta_ts, 1):

	    t = np.linspace(0, 5 * dt, 1000)
	    s = gerar_pulso(t, dt)
	    plt.subplot(3, 1, i)
	    plt.plot(t, s)
	    plt.title(f'Pulso retangular - δt = {dt:.0e} s')
	    plt.grid(True)

	    t, r_t = gerar_canal_multipercurso(N, dt, tau, fc, desvio_doppler, potencia)
	    plt.subplot(3, 1, i)
	    plt.plot(t, np.abs(r_t), 'r', label=f'δt = {dt:.0e} s')
	    plt.title(f'Sinal Recebido, δt = {dt:.0e} s')
	    plt.xlabel("Tempo absoluto - t (s)")
	    plt.ylabel("|r(t)|")
	    plt.grid(True)


	plt.tight_layout()
	plt.show()



# funcao de autocorrelacao

#banda de coerencia
def plot_banda_de_coerencia(tau, potencia, M):

	# Geração de exemplo: função de correlação de frequência 
	k = np.logspace(0, 10, 1000)  # 1 Hz a 10 GHz
	rho_kappa = np.zeros_like(k, dtype=complex)

	for i in range(M):
		rho_kappa += potencia[i] * np.exp( -1j * 2 * np.pi * tau[i] * k )

	rho_kappa_abs = np.abs( rho_kappa ) / np.sum( potencia ) 
	
	# Primeiro ponto abaixo do limiar
	Bc095 = np.argmax(rho_kappa_abs < 0.95)
	Bc090 = np.argmax(rho_kappa_abs < 0.90)

	if Bc095 > 0:
		if Bc090 > 0:
			Bc_095 = k[Bc095]
			Bc_090 = k[Bc090]

			# Plot
			plt.figure(figsize=(10, 5))
			plt.plot(k, rho_kappa_abs)
			plt.axhline(0.95, color='gray', linestyle='-.')
			plt.axhline(0.90, color='gray', linestyle='-.')
			plt.axvline(Bc_095, color='black', linestyle='--', label=f"$B_C(0.95) \\approx {Bc_095 / 10**6:.2f}$ MHz")
			plt.axvline(Bc_090, color='black', linestyle='--', label=f"$B_C(0.90) \\approx {Bc_090 / 10**6:.2f}$ MHz")
			plt.xscale('log')
			plt.xlabel("Desvio de Frequência – $\\kappa$ (Hz)")
			plt.ylabel(r"$|\rho_{TT}(\kappa, 0)|$")
			plt.title(r"$\rho_{TT}(\kappa, 0)$ e Bandas de Coerência")
			plt.grid(True, which='both', linestyle=':', linewidth=0.5)
			plt.legend()
			plt.tight_layout()
			plt.show()
			return
			
		else:
			Bc_095 = k[Bc095]

			# Plot
			plt.figure(figsize=(10, 5))
			plt.plot(k, rho_kappa_abs)
			plt.axhline(0.95, color='gray', linestyle='-.')
			plt.axvline(Bc_095, color='black', linestyle='--', label=f"$B_C(0.95) \\approx {Bc_095 / 10**6:.2f}$ MHz")
			plt.xscale('log')
			plt.xlabel("Desvio de Frequência – $\\kappa$ (Hz)")
			plt.ylabel(r"$|\rho_{TT}(\kappa, 0)|$")
			plt.title(r"$\rho_{TT}(\kappa, 0)$ e Bandas de Coerência")
			plt.grid(True, which='both', linestyle=':', linewidth=0.5)
			plt.legend()
			plt.tight_layout()
			plt.show()
			return
			

	if Bc090 > 0 : 
		Bc_090 = k[Bc090]

		# Plot
		plt.figure(figsize=(10, 5))
		plt.plot(k, rho_kappa_abs)
		plt.axhline(0.90, color='gray', linestyle='-.')
		plt.axvline(Bc_090, color='black', linestyle='--', label=f"$B_C(0.90) \\approx {Bc_090 / 10**6:.2f}$ MHz")
		plt.xscale('log')
		plt.xlabel("Desvio de Frequência – $\\kappa$ (Hz)")
		plt.ylabel(r"$|\rho_{TT}(\kappa, 0)|$")
		plt.title(r"$\rho_{TT}(\kappa, 0)$ e Bandas de Coerência")
		plt.grid(True, which='both', linestyle=':', linewidth=0.5)
		plt.legend()
		plt.tight_layout()
		plt.show()
		return
	
	# Plot
	plt.figure(figsize=(10, 5))
	plt.plot(k, rho_kappa_abs)
	plt.xscale('log')
	plt.xlabel("Desvio de Frequência – $\\kappa$ (Hz)")
	plt.ylabel(r"$|\rho_{TT}(\kappa, 0)|$")
	plt.title(r"$\rho_{TT}(\kappa, 0)$ e Bandas de Coerência")
	plt.grid(True, which='both', linestyle=':', linewidth=0.5)
	#plt.legend()
	plt.tight_layout()
	plt.show()


#tempo de coerencia
def plot_tempo_de_coerencia(desvio_doppler, potencia, M):
	
	sigma = np.logspace(-6, 0, 1000)  # de 1 us até 1 s
	rho_kappa = np.zeros_like(sigma, dtype=complex)

	for i in range(M):
		rho_kappa += potencia[i] * np.exp( 1j * 2 * np.pi * desvio_doppler[i] * sigma )

	rho_kappa_abs = np.abs( rho_kappa ) / np.sum( potencia ) 
	
	# Primeiro ponto abaixo do limiar
	Tc095 = np.argmax(rho_kappa_abs < 0.95)
	Tc090 = np.argmax(rho_kappa_abs < 0.90)


	if Tc095 > 0:
		if Tc090 > 0:
			Tc_095 = sigma[Tc095]
			Tc_090 = sigma[Tc090]

			# Plot
			plt.figure(figsize=(10, 5))
			plt.plot(sigma, rho_kappa_abs)
			plt.axhline(0.95, color='gray', linestyle='-.')
			plt.axhline(0.90, color='gray', linestyle='-.')
			plt.axvline(Tc_095, color='black', linestyle='--', label=f"$T_C(0.95) \\approx {Tc_095 / 10**(-3):.2f}$ ms")
			plt.axvline(Tc_090, color='black', linestyle='--', label=f"$T_C(0.90) \\approx {Tc_090 / 10**(-3):.2f}$ ms")
			plt.xscale('log')
			plt.xlabel("Desvio de Tempo – $\\sigma$ (s)")
			plt.ylabel(r"$|\rho_{TT}(0, \sigma)$ e Tempo de Coerência")
			plt.title(r"$\rho_{TT}(0, \sigma)$ e Tempo de Coerência")
			plt.grid(True, which='both', linestyle=':', linewidth=0.5)
			plt.legend()
			plt.tight_layout()
			plt.show()
			return
			
		else:
			Tc_095 = sigma[Tc095]

			# Plot
			plt.figure(figsize=(10, 5))
			plt.plot(sigma, rho_kappa_abs)
			plt.axhline(0.95, color='gray', linestyle='-.')
			plt.axvline(Tc_095, color='black', linestyle='--', label=f"$T_C(0.95) \\approx {Tc_095 / 10**(-3):.2f}$ ms")
			plt.xscale('log')
			plt.xlabel("Desvio de Tempo – $\\sigma$ (s)")
			plt.ylabel(r"$|\rho_{TT}(0, \sigma)|$")
			plt.title(r"$\rho_{TT}(0, \sigma)$ e Tempo de Coerência")
			plt.grid(True, which='both', linestyle=':', linewidth=0.5)
			plt.legend()
			plt.tight_layout()
			plt.show()
			return
			

	if Tc090 > 0 : 
		Tc_090 = sigma[Tc090]

		# Plot
		plt.figure(figsize=(10, 5))
		plt.plot(sigma, rho_kappa_abs)
		plt.axhline(0.90, color='gray', linestyle='-.')
		plt.axvline(Tc_090, color='black', linestyle='--', label=f"$T_C(0.90) \\approx {Tc_090 / 10**(-3):.2f}$ ms")
		plt.xscale('log')
		plt.xlabel("Desvio de Tempo – $\\sigma$ (s)")
		plt.ylabel(r"$|\rho_{TT}(0, \sigma)|$")
		plt.title(r"$\rho_{TT}(0, \sigma)$ e Tempo de Coerência")
		plt.grid(True, which='both', linestyle=':', linewidth=0.5)
		plt.legend()
		plt.tight_layout()
		plt.show()
		return
	
	# Plot
	plt.figure(figsize=(10, 5))
	plt.plot(sigma, rho_kappa_abs)
	plt.xscale('log')
	plt.xlabel("Desvio de Tempo – $\\sigma$ (s)")
	plt.ylabel(r"$|\rho_{TT}(0, \sigma)|$")
	plt.title(r"$\rho_{TT}(0, \sigma)$ e Tempo de Coerência")
	plt.grid(True, which='both', linestyle=':', linewidth=0.5)
	#plt.legend()
	plt.tight_layout()
	plt.show()




#----------------------------------- Questoes -------------------------------------------------------------------------------------------------------------------------------------------




plot_media_espalhamento_mult(fc)
plot_desvio_espalhamento_mult(fc)

print(f"fator rice achado: {fator_rice} \nfator de rice calculado a partir da expressao: ", potencia[0] / np.sum(potencia[1:] ) )

plot_atraso_de_potencia(tau, potencia)

print(f"o espalhamento de atraso achado foi: {sigma}")
print(f"o espalhamento de atraso calculado pela definicao foi: {sigma_def}")

plot_potencia_angular_azimutal(theta, potencia)
plot_potencia_angular_elevacao(phi, potencia)

plot_direcao_chegada(direcao_chegada)

vrx = 5
numero_coordenada = 1
angulo_azimutal_v = np.pi / 4
angulo_elevacao_v = np.pi / 4
direcao_chegada = np.transpose( direcao_chegada )
desvio_doppler = get_desvio_doppler( vrx, fc, direcao_chegada, angulo_azimutal_v, angulo_elevacao_v )
plot_desvio_doppler( vrx, desvio_doppler, potencia )
vrx = 50
desvio_doppler = get_desvio_doppler( vrx, fc, direcao_chegada, angulo_azimutal_v, angulo_elevacao_v )
plot_desvio_doppler( vrx, desvio_doppler, potencia )

vrx = 5
desvio_doppler = get_desvio_doppler( vrx, fc, direcao_chegada, angulo_azimutal_v, angulo_elevacao_v )
plot_sinal_recebido()
vrx = 50
desvio_doppler = get_desvio_doppler( vrx, fc, direcao_chegada, angulo_azimutal_v, angulo_elevacao_v )
plot_sinal_recebido()

plot_banda_de_coerencia(tau, potencia, N)

vrx = 5
desvio_doppler = get_desvio_doppler( vrx, fc, direcao_chegada, angulo_azimutal_v, angulo_elevacao_v )
plot_tempo_de_coerencia(desvio_doppler, potencia, N)
vrx = 50
desvio_doppler = get_desvio_doppler( vrx, fc, direcao_chegada, angulo_azimutal_v, angulo_elevacao_v )
plot_tempo_de_coerencia(desvio_doppler, potencia, N)
