import React from "react";

export function Slider({ min = 0, max = 1, step = 0.01, defaultValue = [0.5], onValueChange }) {
  const [value, setValue] = React.useState(defaultValue[0]);

  const handleChange = (e) => {
    const newVal = parseFloat(e.target.value);
    setValue(newVal);
    if (onValueChange) onValueChange([newVal]);
  };

  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={handleChange}
      className="w-full"
    />
  );
}
