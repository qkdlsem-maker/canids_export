/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "can_ids_embedded_v2.h"
#include <stdint.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/

/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* F5 results: inspect these volatile variables in the CubeIDE debugger. */
volatile uint32_t f5_system_core_clock_hz = 0;
volatile uint32_t f5_cycles_last = 0;
volatile uint32_t f5_cycles_min = 0xFFFFFFFFu;
volatile uint32_t f5_cycles_max = 0;
volatile uint64_t f5_cycles_sum = 0;
volatile uint32_t f5_measurements = 0;
volatile uint32_t f5_parity_failures = 0;
volatile int f5_last_prediction = -1;
volatile double f5_last_maha_distance = 0.0;

static const double f5_vectors[][CAN_IDS_NUM_FEATURES] = {
  {0.07419379, 16.711567, 1.8825006, 48.596138, 0.07298676, 1.8394724},
  {1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
  {0.5, 10.0, 2.0, 50.0, 0.0, 2.0},
  {1.0, 20.0, 3.0, 127.0, 2.0, 5.0},
  {0.05, 5.0, 1.0, 10.0, -0.5, -1.0},
  {0.8, 2.0, 0.5, 200.0, 4.0, 10.0},
  {0.2, 15.0, 2.5, 60.0, 0.1, 2.0},
  {1.0, 50.0, 4.0, 255.0, 10.0, 20.0}
};
/* Expected outputs were generated from the same frozen V2 C model on host. */
static const int f5_expected[] = {0, 1, 5, 5, 0, 2, 0, 2};
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
/* USER CODE BEGIN PFP */
static void F5_DWT_Init(void);
static uint32_t F5_Predict_Cycles(const double *x, int *prediction);
static void F5_Run_Benchmark(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static void F5_DWT_Init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static uint32_t F5_Predict_Cycles(const double *x, int *prediction)
{
  uint32_t start = DWT->CYCCNT;
  int y = can_ids_predict(x);
  uint32_t end = DWT->CYCCNT;
  *prediction = y;
  return end - start;
}

static void F5_Run_Benchmark(void)
{
  const uint32_t vector_count = (uint32_t)(sizeof(f5_vectors) / sizeof(f5_vectors[0]));
  const uint32_t repeats = 12500u;  /* 12,500 x 8 vectors = 100,000 physical inferences */

  f5_cycles_min = 0xFFFFFFFFu;
  f5_cycles_max = 0u;
  f5_cycles_sum = 0u;
  f5_measurements = 0u;
  f5_parity_failures = 0u;

  /* Warm-up: avoid counting first-call/cache/debugger effects as the only sample. */
  for (uint32_t i = 0; i < vector_count; ++i) {
    f5_last_prediction = can_ids_predict(f5_vectors[i]);
  }

  for (uint32_t r = 0; r < repeats; ++r) {
    for (uint32_t i = 0; i < vector_count; ++i) {
      int prediction = -1;
      uint32_t cycles = F5_Predict_Cycles(f5_vectors[i], &prediction);
      f5_last_prediction = prediction;
      f5_cycles_last = cycles;
      f5_last_maha_distance = maha_distance(f5_vectors[i]);

      if (prediction != f5_expected[i]) {
        ++f5_parity_failures;
      }
      if (cycles < f5_cycles_min) f5_cycles_min = cycles;
      if (cycles > f5_cycles_max) f5_cycles_max = cycles;
      f5_cycles_sum += cycles;
      ++f5_measurements;
    }
  }

  /* LD2 ON = all 100,000 tested decisions matched expected host-C decisions. */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5,
                   (f5_parity_failures == 0u) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  /* USER CODE BEGIN 2 */
  SystemCoreClockUpdate();
  f5_system_core_clock_hz = SystemCoreClock;
  F5_DWT_Init();
  F5_Run_Benchmark();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);

  /*Configure GPIO pins : USART_TX_Pin USART_RX_Pin */
  GPIO_InitStruct.Pin = USART_TX_Pin|USART_RX_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : PA5 */
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
