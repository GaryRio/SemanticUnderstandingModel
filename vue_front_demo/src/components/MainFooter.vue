<template>
  <div class="footer-container">
    <div class="footer-icon">
      <span>客服问答模型</span>
    </div>
  </div>
</template>

<script>
export default {
  name: "MainFooter"
}
</script>

<style scoped>
.footer-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 60px;
  background-color: #e7eefa;
  position: fixed;
  bottom: 0;
  width: 100%;
  z-index: 1000;
  box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
}

.footer-icon {
  font-family: 'Arial', sans-serif;
  font-size: 20px;
  font-weight: bold;
  color: #333;
  text-align: center;
  padding: 0px 15px;
  border-radius: 25px;
  background-color: #dfe8f7;;
  cursor: pointer;
  transition: background-color 0.3s ease, transform 0.3s ease;
  user-select: none;
}

.footer-icon:hover {
  background-color: #cdd5e8;;
  transform: scale(1.05);
}

.footer-icon:active {
  transform: scale(0.95);
}
</style>