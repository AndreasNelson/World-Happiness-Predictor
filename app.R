library(shiny)
library(shinythemes)
library(tidyverse)
library(rsample)
library(recipes)
library(parsnip)
library(workflows)
library(tune)
library(yardstick)
library(ranger)
library(vip)
library(DT)

# Data Loading and Preprocessing
master_whr_data <- readRDS("master_data.rds")

safe_numeric <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}

master_whr_data <- master_whr_data %>%
  mutate(
    life_ladder = safe_numeric(life_ladder),
    log_gdp_per_capita = safe_numeric(log_gdp_per_capita),
    social_support = safe_numeric(social_support),
    healthy_life_expectancy_at_birth = safe_numeric(healthy_life_expectancy_at_birth),
    freedom_to_make_life_choices = safe_numeric(freedom_to_make_life_choices),
    generosity = safe_numeric(generosity),
    perceptions_of_corruption = safe_numeric(perceptions_of_corruption),
    population = safe_numeric(population)
  ) %>%
  filter(!is.na(life_ladder))

# Get list of countries for the dropdown
country_list <- sort(unique(master_whr_data$country_name))

# Model Training

set.seed(123)
data_split <- initial_split(master_whr_data, prop = 0.80, strata = life_ladder)
train_data <- training(data_split)

whr_recipe <- recipe(life_ladder ~ ., data = train_data) %>%
  update_role(country_name, year, new_role = "ID") %>%
  step_impute_median(all_predictors()) %>%
  step_nzv(all_predictors())

# Model Specification
rf_spec <- rand_forest(trees = 100) %>% 
  set_engine("ranger", importance = "permutation") %>%
  set_mode("regression")

# Workflow
rf_workflow <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(whr_recipe)

# Fit Model
rf_fit <- fit(rf_workflow, data = train_data)

# Evaluation Metrics
final_fit_eval <- last_fit(rf_workflow, data_split)
predictions_df <- collect_predictions(final_fit_eval)
metrics_df <- collect_metrics(final_fit_eval)

# Helper functions for UI
safe_min <- function(col) floor(min(col, na.rm = TRUE))
safe_max <- function(col) ceiling(max(col, na.rm = TRUE))
safe_mean <- function(col) mean(col, na.rm = TRUE)

# UI
ui <- navbarPage(
  title = "World Happiness Predictor",
  theme = shinytheme("flatly"),
  
  # Tab 1: Happiness Simulator
  tabPanel("Happiness Simulator",
           sidebarLayout(
             sidebarPanel(
               h4("Configuration"),
               
               # Country Preset Selector
               selectInput("country_selector", "Load Country Data:",
                           choices = c("Global Average (Default)", country_list),
                           selected = "Global Average (Default)"),
               hr(),
               
               h4("Adjust National Indicators"),
               p("Change these values to predict the Life Ladder score."),
               
               sliderInput("gdp", "Log GDP per Capita:",
                           min = safe_min(master_whr_data$log_gdp_per_capita),
                           max = safe_max(master_whr_data$log_gdp_per_capita),
                           value = safe_mean(master_whr_data$log_gdp_per_capita), step = 0.1),
               
               sliderInput("life_exp", "Healthy Life Expectancy:",
                           min = safe_min(master_whr_data$healthy_life_expectancy_at_birth),
                           max = safe_max(master_whr_data$healthy_life_expectancy_at_birth),
                           value = safe_mean(master_whr_data$healthy_life_expectancy_at_birth), step = 1),
               
               sliderInput("social", "Social Support:", min = 0, max = 1, value = 0.8, step = 0.01),
               sliderInput("freedom", "Freedom to Make Life Choices:", min = 0, max = 1, value = 0.7, step = 0.01),
               sliderInput("generosity", "Generosity:", min = -0.3, max = 0.7, value = 0, step = 0.05),
               sliderInput("corruption", "Perceptions of Corruption:", min = 0, max = 1, value = 0.7, step = 0.01),
               
               numericInput("population", "Population:", value = safe_mean(master_whr_data$population))
             ),
             
             mainPanel(
               h3("Predicted Life Ladder Score (0-10)"),
               div(style = "font-size: 60px; color: #2c3e50; font-weight: bold;", textOutput("pred_score")),
               br(),
               # Show which country is currently being simulated
               textOutput("current_sim_label"),
               br(),
               plotOutput("distPlot")
             )
           )
  ),
  
  # Tab 2: Diagnostics
  tabPanel("Model Diagnostics",
           fluidRow(
             column(6, h4("Variable Importance"), plotOutput("vipPlot")),
             column(6, h4("Predicted vs. Actual"), plotOutput("predVsActualPlot"))
           )
  ),
  
  # Tab 3: Data explorer
  tabPanel("Data Explorer",
           fluidRow(
             column(12,
                    h3("Search Country Data"),
                    p("Use the search box on the right to find specific countries."),
                    hr(),
                    DTOutput("country_table")
             )
           )
  ),
  
  # Tab 4: About
  tabPanel("About",
           fluidRow(
             column(8, offset = 2,
                    h3("About this Project"),
                    p("This dashboard predicts national happiness scores using a Random Forest model trained on historical data (2005-2020)."),
                    
                    h4("Step 1: Data Collection & Merging"),
                    p("The final dataset (`master_whr_data`) was constructed by harmonizing two distinct data sources:"),
                    tags$ul(
                      tags$li(strong("World Happiness Report (WHR):"), "Provided subjective well-being metrics (Life Ladder, Social Support, etc.) sourced from the Gallup World Poll."),
                      tags$li(strong("World Bank (WDI):"), "Provided objective economic indicators (Log GDP per capita, Healthy Life Expectancy).")
                    ),
                    p("The merging process involved:"),
                    tags$ol(
                      tags$li("Cleaning column names using `janitor::clean_names()`."),
                      tags$li("Standardizing country names (e.g., mapping 'Turkiye' to 'Turkey' and 'Palestinian Territories' to 'Palestine') to maximize the join rate."),
                      tags$li("Performing an ", strong("inner join"), " on `country_name` and `year`. This ensured that every happiness score in the training set was matched with the correct economic context for that specific year.")
                    ),
                    
                    h4("Step 2: Exploratory Data Analysis (EDA)"),
                    p("Feature selection was driven by EDA correlations, which identified the strongest drivers of happiness:"),
                    tags$ul(
                      tags$li(strong("Strong Positive Correlations:"), "Log GDP (0.78) and Healthy Life Expectancy (0.71)."),
                      tags$li(strong("Moderate Positive Correlations:"), "Social Support (0.72) and Freedom to Make Life Choices (0.54)."),
                      tags$li(strong("Negative Correlation:"), "Perceptions of Corruption (-0.43), indicating that lower corruption is associated with higher happiness."),
                      tags$li(strong("Weak Correlation:"), "Generosity had the weakest link but was retained to capture altruistic behaviors.")
                    ),
                    
                    h4("Modeling Methodology"),
                    p("A Random Forest regression model (100 trees) was chosen to handle the non-linear interactions between these economic and social factors. Missing values in the predictors were handled via median imputation within the `tidymodels` recipe."),
                    
                    h4("Variable Definitions"),
                    tags$ul(
                      tags$li(strong("Life Ladder:"), "The target variable. Responses to the Cantril Ladder question: 'Rate your current life on a scale of 0 to 10'."),
                      tags$li(strong("Log GDP per Capita:"), "Purchasing Power Parity (PPP) at constant 2017 international dollar prices (World Bank)."),
                      tags$li(strong("Social Support:"), "Binary response (0/1) national average: 'If you were in trouble, do you have relatives or friends you can count on?' (Gallup)."),
                      tags$li(strong("Healthy Life Expectancy:"), "Time series of healthy life expectancy at birth (WHO/World Bank)."),
                      tags$li(strong("Freedom to Make Life Choices:"), "National average of satisfaction with freedom of choice (Gallup)."),
                      tags$li(strong("Perceptions of Corruption:"), "National average of responses regarding corruption in government and businesses (Gallup).")
                    )
             )
           )
  )
)

# Server Logic
server <- function(input, output, session) {
  
  # Update sliders based on country selection
  observeEvent(input$country_selector, {
    if(input$country_selector == "Global Average (Default)") {
      # Reset to Averages
      updateSliderInput(session, "gdp", value = safe_mean(master_whr_data$log_gdp_per_capita))
      updateSliderInput(session, "life_exp", value = safe_mean(master_whr_data$healthy_life_expectancy_at_birth))
      updateSliderInput(session, "social", value = 0.8) # Approx mean
      updateSliderInput(session, "freedom", value = 0.7) # Approx mean
      updateSliderInput(session, "generosity", value = 0) # Approx mean
      updateSliderInput(session, "corruption", value = 0.7) # Approx mean
      updateNumericInput(session, "population", value = safe_mean(master_whr_data$population))
    } else {
      # Filter data for selected country
      # Arrange by Year descending to get most recent data
      country_data <- master_whr_data %>%
        filter(country_name == input$country_selector) %>%
        arrange(desc(year)) %>%
        slice(1)
      
      # Update Inputs (Check for NAs and use defaults if missing)
      get_val <- function(val, default) if(is.na(val)) default else val
      
      updateSliderInput(session, "gdp", value = get_val(country_data$log_gdp_per_capita, 9.5))
      updateSliderInput(session, "life_exp", value = get_val(country_data$healthy_life_expectancy_at_birth, 65))
      updateSliderInput(session, "social", value = get_val(country_data$social_support, 0.8))
      updateSliderInput(session, "freedom", value = get_val(country_data$freedom_to_make_life_choices, 0.7))
      updateSliderInput(session, "generosity", value = get_val(country_data$generosity, 0))
      updateSliderInput(session, "corruption", value = get_val(country_data$perceptions_of_corruption, 0.7))
      updateNumericInput(session, "population", value = get_val(country_data$population, 1000000))
    }
  })
  
  # Reactive Prediction
  user_input_data <- reactive({
    tibble(
      country_name = "User Country", year = 2023,
      log_gdp_per_capita = input$gdp, social_support = input$social,
      healthy_life_expectancy_at_birth = input$life_exp, freedom_to_make_life_choices = input$freedom,
      generosity = input$generosity, perceptions_of_corruption = input$corruption, population = input$population
    )
  })
  
  output$pred_score <- renderText({
    req(user_input_data())
    pred <- predict(rf_fit, new_data = user_input_data())
    round(pred$.pred, 2)
  })
  
  output$current_sim_label <- renderText({
    if(input$country_selector == "Global Average (Default)") {
      "Simulating: Global Average"
    } else {
      paste("Simulating based on latest data from:", input$country_selector)
    }
  })
  
  output$distPlot <- renderPlot({
    req(user_input_data())
    pred_val <- predict(rf_fit, new_data = user_input_data())$.pred
    
    ggplot(master_whr_data, aes(x = life_ladder)) +
      geom_density(fill = "#18bc9c", alpha = 0.5) +
      geom_vline(xintercept = pred_val, color = "#e74c3c", size = 2, linetype = "dashed") +
      annotate("text", x = pred_val, y = 0, label = "Your Prediction", 
               vjust = -1, angle = 90, color = "#e74c3c", fontface = "bold") +
      theme_minimal()
  })
  
  output$vipPlot <- renderPlot({
    vip(pull_workflow_fit(rf_fit), num_features = 10) + theme_minimal()
  })
  
  output$predVsActualPlot <- renderPlot({
    predictions_df %>% ggplot(aes(x = life_ladder, y = .pred)) +
      geom_point(alpha = 0.3) + geom_abline(lty = 2, color = "red") + theme_minimal()
  })
  
  output$country_table <- renderDT({
    clean_table <- master_whr_data %>%
      select(Country = country_name, Year = year, 
             `Life Ladder` = life_ladder, `GDP (Log)` = log_gdp_per_capita, 
             `Social Sup` = social_support, `Life Exp` = healthy_life_expectancy_at_birth)
    
    datatable(clean_table, options = list(pageLength = 15, scrollX = TRUE), rownames = FALSE) %>%
      formatRound(columns = c("Life Ladder", "GDP (Log)", "Social Sup", "Life Exp"), digits = 2)
  })
}

shinyApp(ui = ui, server = server)