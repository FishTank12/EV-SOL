import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Set;
import org.json.JSONObject;

public class PowerGridGUI extends JFrame {
    private JComboBox<String> distributorDropdown;
    private JComboBox<String> timeDropdown;
    private JTextField maxCapacityField, currentLoadField, maxGenerationField, currentGenerationField;
    private JButton manualInputButton, dropdownInputButton;
    private JLabel predictionLabel;

    public PowerGridGUI() {
        setTitle("Power Grid AI Prediction");
        setSize(1000, 600);  // Extended size to fit the map
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(null);

        distributorDropdown = new JComboBox<>(getDistributors());
        timeDropdown = new JComboBox<>(getTimes());
        distributorDropdown.setBounds(50, 50, 200, 30);
        timeDropdown.setBounds(50, 100, 200, 30);
        add(distributorDropdown);
        add(timeDropdown);

        maxCapacityField = new JTextField("Max Capacity (kWh)");
        maxCapacityField.setBounds(300, 50, 200, 30);
        currentLoadField = new JTextField("Current Load (kWh)");
        currentLoadField.setBounds(300, 100, 200, 30);
        maxGenerationField = new JTextField("Max Generation Rate (kWh)");
        maxGenerationField.setBounds(300, 150, 200, 30);
        currentGenerationField = new JTextField("Current Generation Rate (kWh)");
        currentGenerationField.setBounds(300, 200, 200, 30);
        add(maxCapacityField);
        add(currentLoadField);
        add(maxGenerationField);
        add(currentGenerationField);

        manualInputButton = new JButton("Predict with Manual Input");
        manualInputButton.setBounds(50, 450, 300, 30);
        manualInputButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                predictWithManualInput();
            }
        });
        add(manualInputButton);

        dropdownInputButton = new JButton("Predict with Dropdown Input");
        dropdownInputButton.setBounds(400, 450, 300, 30);
        dropdownInputButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                predictWithDropdownInput();
            }
        });
        add(dropdownInputButton);

        predictionLabel = new JLabel("Predicted Demand: ");
        predictionLabel.setBounds(200, 500, 400, 30);
        add(predictionLabel);

        // Add the map image
        JLabel mapLabel = new JLabel(new ImageIcon("map.png"));
        mapLabel.setBounds(550, 50, 400, 400);  // Adjust the size and position as needed
        add(mapLabel);

        setVisible(true);
    }

    private String[] getDistributors() {
        Set<String> distributors = new HashSet<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader("../data/final_data.csv"));
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                distributors.add(values[1]);  // Assuming Distributor_ID is the second column
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return distributors.toArray(new String[0]);
    }

    private String[] getTimes() {
        Set<String> times = new HashSet<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader("../data/final_data.csv"));
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                times.add(values[0]);  // Assuming Timestamp is the first column
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return times.toArray(new String[0]);
    }

    private void predictWithManualInput() {
        try {
            JSONObject input = new JSONObject();
            input.put("Max_Capacity_kWh", Double.parseDouble(maxCapacityField.getText()));
            input.put("Current_Load_kWh", Double.parseDouble(currentLoadField.getText()));
            input.put("Max_Generation_Rate_kWh", Double.parseDouble(maxGenerationField.getText()));
            input.put("Current_Generation_Rate_kWh", Double.parseDouble(currentGenerationField.getText()));

            String prediction = callPythonScript(input.toString());
            predictionLabel.setText("Predicted Demand: " + prediction);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void predictWithDropdownInput() {
        try {
            String distributorID = (String) distributorDropdown.getSelectedItem();
            String time = (String) timeDropdown.getSelectedItem();

            JSONObject input = queryData(distributorID, time);

            String prediction = callPythonScript(input.toString());
            predictionLabel.setText("Predicted Demand: " + prediction);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private JSONObject queryData(String distributorID, String time) {
        JSONObject data = new JSONObject();
        try {
            BufferedReader br = new BufferedReader(new FileReader("../data/final_data.csv"));
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                if (values[0].equals(time) && values[1].equals(distributorID)) {
                    data.put("Max_Capacity_kWh", Double.parseDouble(values[6]));
                    data.put("Current_Load_kWh", Double.parseDouble(values[7]));
                    data.put("Max_Generation_Rate_kWh", Double.parseDouble(values[10]));
                    data.put("Current_Generation_Rate_kWh", Double.parseDouble(values[11]));
                    break;
                }
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return data;
    }

    private String callPythonScript(String input) throws Exception {
        ProcessBuilder processBuilder = new ProcessBuilder("python3", "./script.py", input);
        processBuilder.redirectErrorStream(true);
        Process process = processBuilder.start();

        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder result = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            result.append(line);
        }
        System.out.println(result.toString());
        return result.toString();
    }

    public static void main(String[] args) {
        new PowerGridGUI();
    }
}
